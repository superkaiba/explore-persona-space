"""Reduce the #1739 per-context COMPLIANCE DV from the compliance_full raw judge
shards (StrongREJECT-style rubric). Content hygiene: extracts ONLY numeric
score fields via the pipeline's own `_classify_draw`; never prints judge text.

Output: {context_id: {mean, min_over_rollouts, n_rollouts, n_valid_draws}} per
evil rung, written to <DEST>/compliance_percontext.json.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# script-mode puts scripts/ on sys.path[0], not the repo root — add repo root
# so `from scripts.issue1739_compliance_pilot import ...` resolves (#823).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
assert (_REPO_ROOT / "scripts").is_dir(), _REPO_ROOT

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
CF_ROOT = DEST / "issue1739_ctxmap/evil_ood_spread/compliance_full"
RUNGS = ["evil_train", "evil_hh_rlhf", "evil_toxicchat"]
_K_SUFFIX = re.compile(r"_k\d+$")


# Match rollout/draw keys in BOTH the per_persona reduced-summary section
# (``<ctx>_k<NN>``, no score field -> classified parse_fail, ignored) AND the
# all_scores raw-draw section (``<ctx>_k<NN>__<draw>``, carries score/_raw_text).
_ENTRY_KEY = re.compile(r'"([^"]*_k\d+[^"]*)"\s*:\s*\{')


def _balanced_object(text: str, brace_start: int) -> str | None:
    """Return the balanced {...} substring starting at brace_start, or None if truncated."""
    depth = 0
    in_str = False
    esc = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    return None  # truncated (incomplete streaming write)


def _iter_per_persona_entries(text: str):
    """Yield (item_key, parsed_value) for each per_persona rollout entry.

    The shards are pretty-printed but streaming-TRUNCATED mid-object, so a
    whole-file json.loads fails. Entry keys end in ``_k<NN>`` (rollout index);
    each value object is brace-balanced independently, so a truncated tail
    entry simply yields nothing.
    """
    for m in _ENTRY_KEY.finditer(text):
        key = m.group(1)
        obj_txt = _balanced_object(text, m.end() - 1)
        if obj_txt is None:
            continue
        try:
            yield key, json.loads(obj_txt)
        except json.JSONDecodeError:
            continue


def reduce_rung(rung: str) -> dict:
    from scripts.issue1739_compliance_pilot import _classify_draw

    d = CF_ROOT / rung
    files = sorted(d.glob("judge_raw_compliance_full*.jsonl")) + sorted(
        d.glob("judge_raw_compliance_full.json")
    )
    # per rollout item (context_id + _kNN): collect valid draw scores
    per_item: dict[str, list[float]] = {}
    n_entries = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        for key, parsed in _iter_per_persona_entries(text):
            n_entries += 1
            item_id = key.split("__", 1)[0]  # key = "<context_id>_k<NN>"
            score, cls, _ = _classify_draw(parsed)
            if cls == "valid" and score is not None:
                per_item.setdefault(item_id, []).append(float(score))
    # per rollout item -> mean over its valid draws
    item_mean: dict[str, float] = {k: sum(v) / len(v) for k, v in per_item.items() if v}
    # per context -> two-level mean over rollout items (matches per_context_means)
    by_ctx: dict[str, list[float]] = {}
    for item_id, m in item_mean.items():
        ctx = _K_SUFFIX.sub("", item_id)
        by_ctx.setdefault(ctx, []).append(m)
    out = {}
    for ctx, vals in by_ctx.items():
        out[ctx] = {
            "mean": sum(vals) / len(vals),
            "min_over_rollouts": min(vals),
            "n_rollouts": len(vals),
        }
    print(f"  {rung}: {n_entries} entries, {len(item_mean)} rollout-items, {len(out)} contexts")
    return out


def main() -> int:
    result = {}
    for rung in RUNGS:
        result[rung] = reduce_rung(rung)
    allc = {c: v for r in result.values() for c, v in r.items()}
    import numpy as np

    means = np.array([v["mean"] for v in allc.values()])
    mins = np.array([v["min_over_rollouts"] for v in allc.values()])
    print(f"\nTOTAL compliance contexts: {len(allc)}")
    if means.size == 0:
        print("NO valid compliance scores parsed — check the raw-draw schema")
        (DEST / "compliance_percontext.json").write_text(json.dumps(result, indent=1))
        return 0
    print(
        "compliance mean dist: mean %.1f  q50 %.1f q90 %.1f q95 %.1f q99 %.1f max %.1f"
        % (
            means.mean(),
            *[np.percentile(means, q) for q in (50, 90, 95, 99)],
            means.max(),
        )
    )
    for thr in (90, 80, 70, 50):
        print(
            f"  contexts mean>={thr}: {int((means >= thr).sum())}  "
            f"| of which min-over-rollouts>={thr}: {int(((means >= thr) & (mins >= thr)).sum())}"
        )
    print(f"  contexts mean<=5 (failed-compliance): {int((means <= 5).sum())}")
    out = DEST / "compliance_percontext.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
