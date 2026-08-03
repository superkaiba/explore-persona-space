"""Issue #1689 analyzer round-1 digest — per-pair validity-annotated ladder table.

Reads the two committed ladder JSONs (base + instruct, L19) and emits ONE
tidy CSV row per (model, arm, ordered pair) with:

- pair-class labels (framing / identity / provenance / crossed / ...),
- the within-target ceiling + reach bar, the 9 per-rung held-out R^2 values,
- recovery fractions (rung R^2 / ceiling, ceiling > 0 only),
- rung-reached recomputed at thresholds 0.85 / 0.90 / 0.95 with an EXPLICIT
  no-rung-reconciles code (10) — the fit script's `rung_reached_point`
  conflates "rung 9 reconciles" with "nothing reconciles" (both -> 9),
- validity flags:
  * `arm_invalid` — user-cell context arm (the user-cell answer span IS u2,
    and the context arm ends one token past u2, so X_context == Y by
    construction; identity+bias baseline R^2 == 1.0 exactly), and user-cell
    naturalistic prefix arm (prefix boundary == context boundary == answer
    end; both arms degenerate),
  * `degenerate_ceiling` — within-target held-out R^2 <= 0 on the pair's
    common rows (reach bar = -inf -> rung 1 trivially),
- bootstrap CI fields where the pair carries 200 draws (pre-descope subset),
- the matched-capacity null p97.5 (degenerate at 1.0 everywhere — the
  shuffled-answer null zeroes the ceiling, so every null draw reads rung 1).

Output: eval_results/issue_1689/analyzer/pair_digest.csv
Usage: uv run python scripts/issue1689_analyzer_digest.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER_DIR = REPO / "eval_results/issue_1689/ladder"
OUT_DIR = REPO / "eval_results/issue_1689/analyzer"

IDENTITIES = ["user_onpolicy", "user_lmsys", "user_haiku", "assistant", "helios", "wren", "dana"]
FRAMINGS = ["naturalistic", "story", "chat"]
RUNGS = [
    "rung_1_direct",
    "rung_2_ctx_offset",
    "rung_3_ans_offset",
    "rung_4_bias_refit",
    "rung_5_scalar_alpha",
    "rung_6_rotation",
    "rung_7_ctx_reparam",
    "rung_8_ans_reparam",
    "rung_9_full_AMB",
]
NO_RECONCILE = 10  # explicit "no rung reaches the bar" code


def parse_slug(slug: str) -> tuple[str, str]:
    for ident in IDENTITIES:
        if slug.startswith(ident + "_"):
            framing = slug[len(ident) + 1 :]
            assert framing in FRAMINGS, slug
            return ident, framing
    raise ValueError(slug)


def classify(si: str, sf: str, ti: str, tf: str) -> str:
    if si == ti and sf != tf:
        return "user-framing" if si.startswith("user_") else "framing"
    if sf == tf and si != ti:
        if si.startswith("user_") and ti.startswith("user_"):
            return "provenance"
        if si.startswith("user_") or ti.startswith("user_"):
            return "identity-vs-user"
        return "identity"
    return "crossed"


def rung_at(r2s: dict[str, float], ceiling: float, thr: float) -> int:
    """Weakest rung with R^2 >= thr*ceiling; 1 if ceiling <= 0 (bar = -inf);
    NO_RECONCILE when nothing reaches."""
    if ceiling <= 0:
        return 1
    bar = thr * ceiling
    for i, k in enumerate(RUNGS, start=1):
        if r2s[k] >= bar:
            return i
    return NO_RECONCILE


def arm_is_invalid(arm: str, si: str, sf: str, ti: str, tf: str) -> bool:
    src_user, tgt_user = si.startswith("user_"), ti.startswith("user_")
    if arm == "context" and (src_user or tgt_user):
        return True  # X_context == Y for user cells (idbias R^2 == 1.0 exactly)
    if arm == "prefix" and (
        (src_user and sf == "naturalistic") or (tgt_user and tf == "naturalistic")
    ):
        return True  # naturalistic user cells: prefix boundary == answer end
    return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model_tag, short in [
        ("Qwen_Qwen2.5-7B", "base"),
        ("Qwen_Qwen2.5-7B-Instruct", "instruct"),
    ]:
        d = json.loads((LADDER_DIR / f"ladder_{model_tag}_L19.json").read_text())
        for key, pair in d["pairs"].items():
            src, tgt = key.split("__")
            si, sf = parse_slug(src)
            ti, tf = parse_slug(tgt)
            cls = classify(si, sf, ti, tf)
            for arm in ("prefix", "context"):
                a = pair[arm]
                r2s = a["rung_r2s_point"]
                ceiling = a["r2_within_target"]
                bd = a.get("bootstrap_draws", {})
                rec = {
                    "model": short,
                    "arm": arm,
                    "pair": key,
                    "src": src,
                    "tgt": tgt,
                    "src_identity": si,
                    "src_framing": sf,
                    "tgt_identity": ti,
                    "tgt_framing": tf,
                    "cls": cls,
                    "n_common": a["n_common"],
                    "ceiling": ceiling,
                    "reach_bar_90pct": a["reach_bar_90pct"],
                    "rung_point_script": a["rung_reached_point"],
                    "rung085": rung_at(r2s, ceiling, 0.85),
                    "rung090": rung_at(r2s, ceiling, 0.90),
                    "rung095": rung_at(r2s, ceiling, 0.95),
                    "degenerate_ceiling": int(ceiling <= 0),
                    "arm_invalid": int(arm_is_invalid(arm, si, sf, ti, tf)),
                    "n_draws": bd.get("n_draws", 0),
                    "ci_med": bd.get("rung_reached_median"),
                    "ci_lo": bd.get("rung_reached_p025"),
                    "ci_hi": bd.get("rung_reached_p975"),
                    "null_p975": a["matched_capacity_null"]["rung_reached_null_p975"],
                }
                for i, k in enumerate(RUNGS, start=1):
                    rec[f"r2_{i}"] = r2s[k]
                    rec[f"rec_{i}"] = r2s[k] / ceiling if ceiling > 0 else math.nan
                rec["best_r2"] = max(r2s.values())
                rec["best_rec"] = rec["best_r2"] / ceiling if ceiling > 0 else math.nan
                rows.append(rec)

    out = OUT_DIR / "pair_digest.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    import os
    import sys

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. The CSV is written via
    # a closed file handle before return; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
