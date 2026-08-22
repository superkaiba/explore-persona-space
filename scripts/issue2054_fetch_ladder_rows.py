#!/usr/bin/env python
"""Fetch #2054's per-pair ladder JSONs from HF and emit three read-only artifacts.

0 GPU-h, no refits. Everything here reads already-computed ladder rung JSONs.

1. ``/tmp/issue2054_ladder_rows_merged.json`` — the merged (src, tgt, arm) row
   list that ``scripts/issue2054_framing_character_transfer_figs.py`` reads.
2. ``eval_results/issue_2054/analyzer_companions/conv_overlap_check.json`` — the
   per-pair conversation-intersection check: OBSERVED intersection against the
   INDEPENDENT-subset prediction ``|A| * |B| / pool``. The ratio answers whether
   the per-character keep sets are independent draws from the shared
   conversation pool (ratio ~= 1) or share correlated rejections (ratio < 1),
   which decides whether retry waves can lift a cross-character pair to a
   target intersection.
3. ``eval_results/issue_2054/analyzer_companions/chat_to_character_pairs.json``
   — the assistant-CHAT -> story-CHARACTER transfers. The ladder's
   ``assistant_to_character`` class requires source and target to share the
   same STORY form, so these pairs arrive instead via the 2x2 chat anchor
   (assistant chat x inserted, same model), pooled into the 208-edge
   ``twobytwo`` class median and never broken out on their own.

Usage:
  uv run python scripts/issue2054_fetch_ladder_rows.py
"""

from __future__ import annotations

import json
import statistics as st
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HF_REPO = "superkaiba1/explore-persona-space-data"
LADDER_PREFIX = "issue2054_lattice/ladder"
STAGE_DIR = REPO / "data/issue_2054/ladder_stage"
ROWS_OUT = Path("/tmp/issue2054_ladder_rows_merged.json")
COMPANIONS = REPO / "eval_results/issue_2054/analyzer_companions"
CHECK_OUT = COMPANIONS / "conv_overlap_check.json"
CHAT2CHAR_OUT = COMPANIONS / "chat_to_character_pairs.json"
FITS_DIGEST = COMPANIONS / "fits_digest.json"

ASSIST = "conversation_paired_stories_assistant"
RUNGS = [
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
]


def _branch_blob(rel: str) -> str:
    """Read a path's blob from ``origin/issue-2054`` (the round's own branch)."""
    return subprocess.run(
        ["git", "show", f"origin/issue-2054:{rel}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _pool_size() -> int:
    """Conversations in the shared fold map — the draw every cell samples from.

    Read from the BRANCH: ``main``'s committed copy is the stale 2026-08-04
    single-variant smoke map (n_conv_ids 1,761, ``variants: [char_helios]``),
    while the production map (26,889 conversations, all five variants) is the
    2026-08-06 blob on ``origin/issue-2054``. Using main's copy silently
    understates the pool by ~15x and makes every overlap ratio meaningless.
    """
    fm = json.loads(_branch_blob("eval_results/issue_2054/shared_fold_map.json"))
    n = int(fm["n_conv_ids"])
    variants = fm.get("variants") or []
    if n < 20000 or len(variants) < 5:
        raise SystemExit(
            f"refusing a non-production fold map: n_conv_ids={n:,}, variants={variants}"
        )
    return n


def _cell_rows() -> dict[str, int]:
    """Per-cell realized row count (one row per conversation) from the fits digest.

    ``analyzer_companions/`` is committed on branch ``issue-2054`` only, so fall
    back to reading the blob out of the branch when the worktree copy is absent.
    """
    text = (
        FITS_DIGEST.read_text()
        if FITS_DIGEST.exists()
        else _branch_blob(str(FITS_DIGEST.relative_to(REPO)))
    )
    d = json.loads(text)
    return {r["cell"]: int(r["inter"]) for r in d["rows"]}


def _read_rows() -> list[dict]:
    """Parse every staged ladder JSON into a flat (src, tgt, arm) row list.

    The per-pair payload lives under ``arm_report`` (status / n_intersection /
    pooled); the TOP-level ``pooled`` key is an empty placeholder, so reading it
    yields zero rows.
    """
    rows: list[dict] = []
    for i, p in enumerate(sorted(STAGE_DIR.rglob("*.json"))):
        d = json.loads(p.read_text())
        ar = d.get("arm_report") or {}
        if ar.get("status") != "ok":
            continue
        pooled = ar.get("pooled") or {}
        rung_means = {k: (pooled.get(k) or {}).get("r2_transfer_mean") for k in RUNGS}
        if any(v is None for v in rung_means.values()):
            continue
        rows.append(
            {
                "src": d["source"],
                "tgt": d["target"],
                "arm": d["arm"],
                "n": int(ar["n_intersection"]),
                "ceiling": d["target_ceiling"],
                "rungs": rung_means,
            }
        )
        if (i + 1) % 200 == 0:
            print(f"[read] {i + 1} files parsed", flush=True)
    return rows


def _overlap_check(rows: list[dict]) -> dict:
    """Observed vs independent-subset-predicted conversation intersection."""
    pool, sizes = _pool_size(), _cell_rows()
    checks = []
    for r in rows:
        if r["arm"] != "context":
            continue
        a, b = sizes.get(r["src"]), sizes.get(r["tgt"])
        if not a or not b:
            continue
        pred = a * b / pool
        checks.append(
            {
                "src": r["src"],
                "tgt": r["tgt"],
                "observed": r["n"],
                "predicted_independent": round(pred, 1),
                "ratio": round(r["n"] / pred, 4),
                "cross_character": r["src"].startswith("char_") and r["tgt"].startswith("char_"),
            }
        )
    xchar = [c["ratio"] for c in checks if c["cross_character"]]
    other = [c["ratio"] for c in checks if not c["cross_character"]]
    return {
        "pool_conversations": pool,
        "n_pairs_checked": len(checks),
        "cross_character": {
            "n": len(xchar),
            "median_ratio": round(st.median(xchar), 4) if xchar else None,
            "min_ratio": round(min(xchar), 4) if xchar else None,
            "max_ratio": round(max(xchar), 4) if xchar else None,
        },
        "other_pairs": {
            "n": len(other),
            "median_ratio": round(st.median(other), 4) if other else None,
        },
        "reading": (
            "ratio = observed intersection / independent-subset prediction "
            "(|A|*|B|/pool). ~1.0 => per-cell keep sets are independent draws, so "
            "retry waves lift pairwise overlap as the independent model predicts; "
            "<1 => correlated rejection (some conversations fail for every "
            "character), so retries buy less than modeled."
        ),
        "pairs": checks,
    }


def _chat_to_character(rows: list[dict]) -> dict:
    """Assistant-CHAT source -> story-CHARACTER target transfers, broken out."""
    out = []
    for r in rows:
        s, t = r["src"], r["tgt"]
        if not s.startswith(f"{ASSIST}__") or not t.startswith("char_"):
            continue
        # cell key = <variant>__<condition>__<form>__<model>
        s_cond, s_form = s.split("__")[1], s.split("__")[2]
        t_cond, t_form, t_model = t.split("__")[1], t.split("__")[2], t.split("__")[3]
        if s_form != "chat" or t_form == "chat":
            continue
        out.append(
            {
                "src": s,
                "tgt": t,
                "arm": r["arm"],
                "source_condition": s_cond,
                "target_condition": t_cond,
                "target_form": t_form,
                "model": t_model,
                "n_intersection": r["n"],
                "target_ceiling": r["ceiling"],
                "rungs": r["rungs"],
                "ratio_of_ceiling": {
                    k: (round(v / r["ceiling"], 4) if r["ceiling"] else None)
                    for k, v in r["rungs"].items()
                },
            }
        )
    ctx = [p for p in out if p["arm"] == "context"]
    return {
        "what": (
            "assistant chat-template map -> story-character map. Enumerated only "
            "via the ladder's 2x2 chat anchor (assistant chat x INSERTED, same "
            "model), so an on-policy chat source has no such pair by construction."
        ),
        "n_pairs": len(out),
        "n_context_arm": len(ctx),
        "source_conditions_present": sorted({p["source_condition"] for p in out}),
        "median_r2_per_rung_context": (
            {k: round(st.median([p["rungs"][k] for p in ctx]), 4) for k in RUNGS} if ctx else None
        ),
        "median_ratio_of_ceiling_context": (
            {k: round(st.median([p["ratio_of_ceiling"][k] for p in ctx]), 4) for k in RUNGS}
            if ctx
            else None
        ),
        "pairs": out,
    }


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    paths = hub.stage_hub_prefix(HF_REPO, LADDER_PREFIX, STAGE_DIR, max_workers=6)
    print(f"[stage] {len(paths)} ladder files under {STAGE_DIR}", flush=True)

    rows = _read_rows()
    ROWS_OUT.write_text(json.dumps(rows))
    print(f"[rows] wrote {len(rows)} rows -> {ROWS_OUT}", flush=True)

    COMPANIONS.mkdir(parents=True, exist_ok=True)

    check = _overlap_check(rows)
    CHECK_OUT.write_text(json.dumps(check, indent=1))
    print(json.dumps({k: v for k, v in check.items() if k != "pairs"}, indent=1), flush=True)

    c2c = _chat_to_character(rows)
    CHAT2CHAR_OUT.write_text(json.dumps(c2c, indent=1))
    print(json.dumps({k: v for k, v in c2c.items() if k != "pairs"}, indent=1), flush=True)
    print(f"[done] {CHECK_OUT}  {CHAT2CHAR_OUT}", flush=True)


if __name__ == "__main__":
    main()
