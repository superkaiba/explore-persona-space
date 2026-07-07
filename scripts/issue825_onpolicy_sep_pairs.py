"""Issue #825 `onpolicy-separator-control` G2b: on-policy separator pairs.

#931 Arm-C recipe VERBATIM (same helpers: ``tokenize_with_offsets``,
``sentence_bounds``, ``covering_token_span`` / ``inner_token_span``, the
anchor-eligibility ladder, per-article seeded RNG ``BUILD_SEED +
article_idx``, ``PairSpec.validate``) EXCEPT the two continuation-region
constraints (plan section 4 G2b — the one changed variable is span-text
PROVENANCE):

  1. the anchor's sentence-final punctuation char offset must be
     ``>= len(prefix_text)`` (anchors restricted to the model-written
     continuation region), and
  2. the preceding-sentence span is CLAMPED to start no earlier than the
     continuation start; a straddling first-sentence span shrinking below
     ``ARMC_PREV_MIN_TOKENS`` DROPS the pair (counted per class, never
     silent — consistent with #931's joint eligibility ladder).

Full text per (article, wave) window = decoded pinned prefix (256 tokens
wave 1 / 512 tokens wave 2) + the model's own raw-text continuation,
re-tokenized through the #931 code path (re-tokenization convention; the
generation-ids seam mismatch is reported as a diagnostic, non-gating).

Outputs (CONSUMER-EXACT filenames so ``issue931_extract_store`` +
``issue931_fit_cells`` run verbatim), under ``<out-data-dir>/pairs/``:
  articles_armC.jsonl   one row per (article, wave) window (re-tokenized ids)
  pairs_armC.jsonl      one PairSpec per kept anchor
  pairs_meta.json       drop accounting + separator-frequency / span-length /
                        anchor-position distributions vs the exogenous pairs

``window_id`` = ``wiki:NNNNN`` (wave 1) / ``wiki:NNNNN:w2`` (wave 2);
``group_id`` = the ARTICLE either way (<= 600 distinct groups asserted).
Yield floor: target 3,600 pairs per model — a shortfall is REPORTED
(``realized_n`` in every output), never padded.

CLI:
  uv run python scripts/issue825_onpolicy_sep_pairs.py \
      --articles <pinned articles_armC.jsonl> \
      --continuations <gen continuations.jsonl> \
      --out-data-dir data/issue_825/onpolicy_sep/<model> \
      --exogenous-articles <pinned> --exogenous-pairs <pinned> --model <m>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps bind before numpy import

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue825_onpolicy_sep_pairs.py"

# Wave geometry (plan section 11): prefix 256/512 tokens, continuation budget
# 768/512 tokens — total window <= 1024 = ARMC_ARTICLE_CAP_TOKENS (extraction
# shape parity with the exogenous arm; structural, not tuned).
PREFIX_TOKENS = {1: 256, 2: 512}
CONTINUATION_MAX_TOKENS = {1: 768, 2: 512}
WAVE2_MIN_ELIGIBLE = common.ARMC_MAX_ANCHORS_PER_ARTICLE  # 6
TARGET_N = common.ARMC_N_ARTICLES * common.ARMC_MAX_ANCHORS_PER_ARTICLE  # 3600


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--articles", type=Path, required=True, help="pinned articles_armC.jsonl")
    ap.add_argument("--continuations", type=Path, required=True, help="gen continuations.jsonl")
    ap.add_argument("--out-data-dir", type=Path, required=True)
    ap.add_argument("--exogenous-articles", type=Path, default=None)
    ap.add_argument("--exogenous-pairs", type=Path, default=None)
    ap.add_argument("--model", type=str, required=True, choices=("base", "instruct"))
    ap.add_argument("--max-anchors", type=int, default=common.ARMC_MAX_ANCHORS_PER_ARTICLE)
    ap.add_argument("--target-n", type=int, default=TARGET_N)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    """Newline-split JSONL read (never splitlines(); gotchas.md JSONL rule)."""
    assert path.exists(), f"missing input: {path}"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def article_idx_of(article_id: str) -> int:
    """'wiki:NNNNN' -> NNNNN (the #931 per-article RNG key)."""
    assert article_id.startswith("wiki:"), article_id
    return int(article_id.split(":")[1])


# ---------------------------------------------------------------------------
# The anchor-eligibility ladder (#931 verbatim + the two G2b constraints)
# ---------------------------------------------------------------------------


def build_wave(tokenizer, article: dict, continuation_text: str, wave: int) -> dict:
    """One (article, wave) window: re-tokenize prefix+continuation, run the
    ladder. Returns {ids, offsets, prefix_text, eligible, counters, ...};
    ``eligible`` entries are (t, span_lo, span_hi, ps_lo, ps_hi, sep_char)."""
    assert wave in PREFIX_TOKENS, wave
    src_ids = list(article["input_ids"])
    assert len(src_ids) >= common.ARMC_ARTICLE_MIN_TOKENS, (article["window_id"], len(src_ids))
    prefix_text = tokenizer.decode(src_ids[: PREFIX_TOKENS[wave]])
    full_text = prefix_text + continuation_text
    ids, offsets = common.tokenize_with_offsets(tokenizer, full_text)
    # Cap the re-tokenized window at the #931 article cap (extraction parity).
    cap = min(len(ids), common.ARMC_ARTICLE_CAP_TOKENS)
    ids, offsets = ids[:cap], offsets[:cap]
    full_text = full_text[: int(offsets[-1, 1])]
    bounds = common.sentence_bounds(full_text)
    cont_char_start = len(prefix_text)
    # First token fully inside the continuation region (the clamp floor);
    # cont_char_start beyond the capped text => zero-eligible window.
    cont_tok_lo = int(np.searchsorted(offsets[:, 0], cont_char_start, side="left"))

    counters: Counter[str] = Counter()
    anchors: list[tuple[int, int, str]] = []  # (token, sentence_char_start, sep_char)
    for s, e in bounds:
        seg = full_text[s:e].rstrip()
        if not seg or seg[-1] not in ".!?":
            continue
        pi = s + len(seg) - 1
        lo, hi = common.covering_token_span(offsets, pi, pi + 1)
        if hi - lo != 1:
            counters["anchor_multi_token"] += 1
            continue
        t = lo
        tok_text = full_text[int(offsets[t, 0]) : int(offsets[t, 1])].strip()
        if tok_text not in (".", "!", "?"):
            counters["anchor_token_not_punct"] += 1
            continue
        if pi < cont_char_start:
            # G2b constraint 1: prefix-region anchors are OUT (counted). All
            # excluded anchors positionally precede every kept anchor, so the
            # next-anchor span boundaries below are unchanged vs #931.
            counters["anchor_in_prefix_region"] += 1
            continue
        anchors.append((t, s, seg[-1]))

    eligible: list[tuple[int, int, int, int, int, str]] = []
    if len(anchors) < 2:
        counters["window_lt2_anchors"] += 1
    else:
        anchor_positions = [a for a, _, _ in anchors]
        for j, (t, sent_start, sep_char) in enumerate(anchors):
            nxt = anchor_positions[j + 1] if j + 1 < len(anchors) else len(ids)
            span_lo, span_hi = t + 1, nxt
            if not (common.ARMC_SPAN_MIN <= span_hi - span_lo <= common.ARMC_SPAN_MAX):
                counters["span_len_out_of_range"] += 1
                continue
            ps_lo, ps_hi = common.inner_token_span(offsets, sent_start, int(offsets[t, 0]))
            ps_hi = min(ps_hi, t)
            ps_lo = max(ps_lo, ps_hi - common.ARMC_PREV_CAP_TOKENS)  # keep LAST <=96 tokens
            clamped_lo = max(ps_lo, cont_tok_lo)  # G2b constraint 2
            if ps_hi - clamped_lo < common.ARMC_PREV_MIN_TOKENS:
                key = "prev_span_clamped_below_min" if clamped_lo > ps_lo else "prev_span_below_min"
                counters[key] += 1
                continue
            eligible.append((t, span_lo, span_hi, clamped_lo, ps_hi, sep_char))
    return {
        "ids": ids,
        "offsets": offsets,
        "prefix_text": prefix_text,
        "cont_char_start": cont_char_start,
        "eligible": eligible,
        "counters": counters,
        "wave": wave,
    }


def count_eligible(tokenizer, article: dict, continuation_text: str, wave: int = 1) -> int:
    """Eligible-anchor count (pre-cap) — the gen script's wave-2 trigger read."""
    return len(build_wave(tokenizer, article, continuation_text, wave)["eligible"])


def select_article_pairs(
    article_id: str, wave_results: dict[int, dict], max_anchors: int
) -> list[tuple[int, common.PairSpec]]:
    """<= max_anchors kept anchors per ARTICLE across waves (wave 1 first;
    wave 2 tops up the remaining budget). Seeded per-article RNG (#931 parity:
    ``default_rng(BUILD_SEED + article_idx)``); PairSpec.validate on every pair."""
    art_idx = article_idx_of(article_id)
    kept: list[tuple[int, common.PairSpec]] = []
    budget = max_anchors
    for wave in (1, 2):
        wr = wave_results.get(wave)
        if wr is None or budget <= 0 or not wr["eligible"]:
            continue
        art_rng = np.random.default_rng(common.BUILD_SEED + art_idx)
        take = art_rng.choice(
            len(wr["eligible"]), size=min(budget, len(wr["eligible"])), replace=False
        )
        window_id = article_id if wave == 1 else f"{article_id}:w2"
        for j in sorted(int(v) for v in take):
            t, span_lo, span_hi, ps_lo, ps_hi, sep_char = wr["eligible"][j]
            pair = common.PairSpec(
                row_id=f"{window_id}:a{t}",
                group_id=article_id,
                char_id="sep",
                c_span=(ps_lo, ps_hi),
                t_spans=[(span_lo, span_hi)],
                ctx_span=(ps_lo, ps_hi),
                meta={
                    "window_id": window_id,
                    "anchor_pos": int(t),
                    "wave": wave,
                    "sep_char": sep_char,
                },
            )
            pair.validate(
                len(wr["ids"]), min_c=common.ARMC_PREV_MIN_TOKENS, min_t=common.ARMC_SPAN_MIN
            )
            assert pair.meta["anchor_pos"] < span_lo, "anchor must precede its span"
            kept.append((wave, pair))
        budget -= len(take)
    return kept


# ---------------------------------------------------------------------------
# Diagnostics: seam mismatch + distribution summaries vs the exogenous pairs
# ---------------------------------------------------------------------------


def seam_mismatch(article: dict, cont_row: dict, retok_ids: list[int], wave: int) -> dict:
    """Re-tokenized ids vs generation-time ids (prefix ids + generation token
    ids), compared over the capped window. Diagnostic only, NON-gating."""
    gen_tail = cont_row.get("continuation_token_ids") or []
    gen_ids = list(article["input_ids"][: PREFIX_TOKENS[wave]]) + list(gen_tail)
    gen_ids = gen_ids[: len(retok_ids)]
    m = min(len(retok_ids), len(gen_ids))
    first_div = next((i for i in range(m) if retok_ids[i] != gen_ids[i]), m)
    exact = first_div == m and len(gen_ids) == len(retok_ids)
    return {"exact": bool(exact), "first_divergence": int(first_div), "n_retok": len(retok_ids)}


def dist_summary(values: list[float | int]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "p25": float(np.quantile(v, 0.25)),
        "p50": float(np.quantile(v, 0.50)),
        "p75": float(np.quantile(v, 0.75)),
        "max": float(v.max()),
        "values": [float(x) for x in v],
    }


def pair_stats(pairs: list[common.PairSpec]) -> dict:
    """Separator-type frequencies + span-length + anchor-position distributions."""
    seps = Counter(p.meta.get("sep_char", "?") for p in pairs)
    return {
        "separator_frequencies": dict(seps),
        "span_length": dist_summary([p.t_spans[0][1] - p.t_spans[0][0] for p in pairs]),
        "anchor_position": dist_summary([int(p.meta["anchor_pos"]) for p in pairs]),
    }


def exogenous_stats(tokenizer, articles_path: Path, pairs_path: Path) -> dict:
    """Recompute the same nuisance-covariate stats from the PINNED exogenous
    pair files (sep char decoded from the pinned article ids at anchor_pos)."""
    arts = {r["window_id"]: r for r in read_jsonl(articles_path)}
    pairs = [common.PairSpec.from_dict(d) for d in read_jsonl(pairs_path)]
    for p in pairs:
        ids = arts[p.meta["window_id"]]["input_ids"]
        p.meta["sep_char"] = tokenizer.decode([ids[int(p.meta["anchor_pos"])]]).strip()
    out = pair_stats(pairs)
    out["n_pairs"] = len(pairs)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"[i825-ops-pairs] wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    tokenizer = common.get_tokenizer()
    articles = {r["window_id"]: r for r in read_jsonl(args.articles)}
    cont_rows = read_jsonl(args.continuations)
    by_article: dict[str, dict[int, dict]] = {}
    for r in cont_rows:
        by_article.setdefault(r["window_id"], {})[int(r["wave"])] = r

    articles_out: list[dict] = []
    pairs_out: list[common.PairSpec] = []
    counters: Counter[str] = Counter()
    seam: list[dict] = []
    n_wave2_windows = 0
    for article_id in sorted(by_article, key=article_idx_of):
        art = articles[article_id]
        waves = by_article[article_id]
        assert 1 in waves, f"{article_id}: wave-1 continuation missing"
        wave_results: dict[int, dict] = {}
        for wave, row in sorted(waves.items()):
            wr = build_wave(tokenizer, art, row["continuation"], wave)
            wave_results[wave] = wr
            counters.update(wr["counters"])
            seam.append(
                {"window_id": article_id, "wave": wave, **seam_mismatch(art, row, wr["ids"], wave)}
            )
        kept = select_article_pairs(article_id, wave_results, args.max_anchors)
        waves_with_pairs = sorted({w for w, _ in kept})
        for wave in waves_with_pairs:
            wr = wave_results[wave]
            window_id = article_id if wave == 1 else f"{article_id}:w2"
            if wave == 2:
                n_wave2_windows += 1
            articles_out.append(
                {
                    "window_id": window_id,
                    "novel_id": article_id,
                    "window_idx": wave,
                    "title": art.get("title", ""),
                    "input_ids": list(wr["ids"]),
                }
            )
        pairs_out.extend(p for _, p in kept)

    groups = sorted({p.group_id for p in pairs_out})
    assert len(groups) <= common.ARMC_N_ARTICLES, (len(groups), common.ARMC_N_ARTICLES)
    realized_n = len(pairs_out)
    shortfall = realized_n < args.target_n
    mismatch_rate = sum(1 for s in seam if not s["exact"]) / len(seam) if seam else float("nan")

    pairs_dir = args.out_data_dir / "pairs"
    _write_jsonl(pairs_dir / "articles_armC.jsonl", articles_out)
    _write_jsonl(pairs_dir / "pairs_armC.jsonl", [p.to_dict() for p in pairs_out])

    meta = {
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, realized_n),
        "followup_label": "onpolicy-separator-control",
        "model": args.model,
        "realized_n": realized_n,
        "target_n": args.target_n,
        "shortfall": bool(shortfall),
        "n_articles_with_pairs": len(groups),
        "n_windows": len(articles_out),
        "n_wave2_windows": n_wave2_windows,
        "drop_counters": dict(counters),
        "onpolicy_stats": pair_stats(pairs_out),
        "exogenous_stats": (
            exogenous_stats(tokenizer, args.exogenous_articles, args.exogenous_pairs)
            if args.exogenous_articles and args.exogenous_pairs
            else None
        ),
        "seam_token_mismatch": {
            "rate": mismatch_rate,
            "n_windows": len(seam),
            "note": "re-tokenization convention (#931 code path); diagnostic, NON-gating",
            "per_window": seam,
        },
        "wave_geometry": {
            "prefix_tokens": PREFIX_TOKENS,
            "continuation_max_tokens": CONTINUATION_MAX_TOKENS,
            "cap_tokens": common.ARMC_ARTICLE_CAP_TOKENS,
        },
    }
    common.write_json(pairs_dir / "pairs_meta.json", meta)
    print(
        f"[i825-ops-pairs] model={args.model} realized_n={realized_n}/{args.target_n} "
        f"(shortfall={shortfall}) groups={len(groups)} wave2_windows={n_wave2_windows} "
        f"seam_mismatch_rate={mismatch_rate:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
