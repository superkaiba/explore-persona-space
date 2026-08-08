"""#1482: blinded A/B packets for the two feature-level qualitative reads.

TWO packet sets, each a pair of group files a reader can compare with no idea
which is which:

  extremes   best-100 vs worst-100 predicted features (set A, the global R^2
             tails of the feature-extremes round)
  side       context-only vs answer-only features (side_class 0 vs 2)

Each item is the feature's DESCRIPTION only -- the judged `label` for the
extremes set, the banked `description` for the side set. No R^2, no side_class,
no activity, no ordering signal: items are shuffled by a fixed hash of feat_id
so rank carries nothing.

THE KEY IS FROZEN. An existing key.json is reused verbatim and only deleting it
re-draws. Re-drawing between a reader opening a packet and the key being read
silently relabels the packets -- that happened on the context round (2026-08-04)
and briefly produced a wrong score.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
EXTREMES = REPO / "eval_results/issue_1482/feature_extremes/extremes.json"
SIDE = REPO / "eval_results/issue_1482/side_specific/side_specific_features.json"
CTX_DESC = REPO / (
    "eval_results/issue_1482/context_side_labels/descriptions_context_side.jsonl"
)
BLIND = REPO / "data/issue_1482/feature_blind"
N_SIDE = 100  # per side, matched, so neither group is bigger by construction


def shuffled(rows: list[dict], key: str = "feat_id") -> list[dict]:
    """Deterministic order that carries no rank signal."""
    return sorted(rows, key=lambda r: int(r[key]) * 2654435761 % 2**32)


def write_pair(name: str, pos: list[str], neg: list[str], pos_kind: str, neg_kind: str) -> dict:
    """Write group A/B files for one packet set, reusing a frozen key if present."""
    BLIND.mkdir(parents=True, exist_ok=True)
    kp = BLIND / "key.json"
    key = json.loads(kp.read_text()) if kp.exists() else {}
    if name in key:
        print(f"[blind] {name}: key FROZEN from {kp} (delete it to re-draw)")
    else:
        a_is_pos = secrets.randbelow(2) == 0
        key[name] = {
            "A": pos_kind if a_is_pos else neg_kind,
            "B": neg_kind if a_is_pos else pos_kind,
        }
    if not pos or not neg:
        raise SystemExit(
            f"{name}: refusing to write a degenerate packet pair "
            f"({pos_kind}={len(pos)}, {neg_kind}={len(neg)}). A group with no items is not a "
            "blindable comparison -- fix the selection or the evidence source first."
        )
    by_kind = {pos_kind: pos, neg_kind: neg}
    for g in ("A", "B"):
        items = by_kind[key[name][g]]
        # HEADER LEAK, fixed 2026-08-04: this line used to interpolate `name`
        # ("extremes" / "side"), so the packet announced its own criterion family.
        # The side reader used exactly that to rule out a top-vs-random design:
        # "both file headers are titled 'side — group A/B' ... implying two sides
        # of one axis". A blinded packet must not name the axis it was split on.
        # Also neutral on WHAT the items are: the earlier wording said "feature
        # descriptions", which told the reader these were model features. A blinded
        # packet states the count and nothing else.
        lines = [f"# Group {g}", "", f"{len(items)} items.", ""]
        lines += [f"{i}. {t}" for i, t in enumerate(items, 1)]
        (BLIND / f"{name}_group{g}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[blind] wrote {name}_group{g}.md  ({len(items)} items)")
    kp.write_text(json.dumps(key, indent=1))
    return key


def main() -> None:
    d = json.loads(EXTREMES.read_text())
    feats = d["features"]
    best = [f for f in shuffled(feats) if f.get("a_best")]
    worst = [f for f in shuffled(feats) if f.get("a_worst")]
    print(f"[blind] extremes: best={len(best)} worst={len(worst)}")
    write_pair(
        "extremes",
        [f["label"] for f in best if f.get("label")],
        [f["label"] for f in worst if f.get("label")],
        "best",
        "worst",
    )

    s = json.loads(SIDE.read_text())
    sf = s["features"]
    # side_class is a STRING here ('context_only'/'answer_only'), NOT the int code
    # (0/1/2/-1) the npz-backed scripts use. Do not copy that encoding across.
    #
    # Descriptions come from TWO files because they come from two evidence sides:
    #   answer_only  -> side_specific_features.json  (#1773 ANSWER-side windows)
    #   context_only -> descriptions_context_side.jsonl (#1482 CONTEXT-side windows,
    #                   1,653 / 1,654 described, 0 drops -- commit d2eb71d9d7)
    # A context-only feature has no answer-side windows by construction, so this
    # split is forced. The meta's EVIDENCE-SIDE CAVEAT is explicit that the two
    # sets are NOT same-instrument: a discriminator a blinded reader finds may be
    # an artifact of the evidence side rather than a difference in feature KIND.
    # That confound is named in the read's brief and carried with its result.
    ctx_desc = {}
    for line in CTX_DESC.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("description"):
                ctx_desc[int(r["feat_id"])] = r["description"]
    ctx = [
        f for f in shuffled(sf)
        if f.get("side_class") == "context_only" and int(f["feat_id"]) in ctx_desc
    ]
    ans = [
        f for f in shuffled(sf)
        if f.get("side_class") == "answer_only" and f.get("description")
    ]
    print(f"[blind] side: context_only={len(ctx)} answer_only={len(ans)}")
    n = min(N_SIDE, len(ctx), len(ans))
    write_pair(
        "side",
        [ctx_desc[int(f["feat_id"])] for f in ctx[:n]],
        [f["description"] for f in ans[:n]],
        "context_only",
        "answer_only",
    )
    print(f"[blind] side packets matched at n={n} per group")


if __name__ == "__main__":
    main()
