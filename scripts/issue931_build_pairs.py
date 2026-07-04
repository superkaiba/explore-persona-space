"""Issue #931 P0: PDNC staging + Arm-A pair construction + G0 gate + Arm-C
pairs + Arm-B prompt battery.

CPU-only (tokenizer, no model). Runs on the VM pre-provision AND on the pod
(gitignored data/ does not travel with the clone — the dispatcher re-runs it).

Outputs (under --data-dir, default data/issue_931):
  pairs/windows_armA.jsonl   one row per Arm-A novel window (token ids)
  pairs/pairs_armA.jsonl     one PairSpec per (window, character)
  pairs/articles_armC.jsonl  one row per WikiText article (token ids)
  pairs/pairs_armC.jsonl     one PairSpec per separator anchor
  pairs/prompt_battery.json  the 1,200 Arm-B story prompts
  pairs/pairs_meta.json      drops / mismatch rates / pins / span-text sample
G0 gate JSON -> --out-dir (default eval_results/issue_931)/g0_gate.json.

CLI:
  uv run python scripts/issue931_build_pairs.py \
      [--pdnc-dir data/issue_931/pdnc] [--pdnc-sha <sha>] [--max-novels N]
      [--n-articles 600] [--max-anchors 6] [--data-dir ...] [--out-dir ...]
      [--skip-armc] [--smoke]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_build_pairs.py"

TEXT_CANDIDATES = ("novel_text.txt", "text.txt", "novel.txt")
QUOTE_CANDIDATES = ("quotation_info.csv", "quotations.csv", "quote_info.csv")
CHAR_CANDIDATES = ("character_info.csv", "characters.csv")

# Speaker labels that are not a single narrating character (groups / unknown /
# narrator entries are excluded per plan section 4.0).
NON_CHARACTER_SPEAKER_RE = re.compile(r"(?i)^\s*(unknow|narrator|crowd|group|voices|_)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdnc-dir", type=Path, default=Path("data/issue_931/pdnc"))
    ap.add_argument(
        "--pdnc-sha",
        type=str,
        default=common.PDNC_SHA,
        help="pin the PDNC checkout to this SHA (default: the committed PDNC_SHA pin — "
        "EVERY path, including the dispatcher's, checks out the pin unconditionally)",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--max-novels", type=int, default=0, help="0 = all novels")
    ap.add_argument("--window-tokens", type=int, default=common.WINDOW_TOKENS)
    ap.add_argument("--max-pairs", type=int, default=common.MAX_ARMA_PAIRS)
    ap.add_argument("--n-articles", type=int, default=common.ARMC_N_ARTICLES)
    ap.add_argument("--max-anchors", type=int, default=common.ARMC_MAX_ANCHORS_PER_ARTICLE)
    ap.add_argument("--skip-armc", action="store_true")
    ap.add_argument("--skip-pdnc-clone", action="store_true", help="use an existing --pdnc-dir")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny slice; numeric G0 gate values recorded but not binding",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# PDNC staging
# ---------------------------------------------------------------------------


def stage_pdnc(pdnc_dir: Path, sha: str | None, *, skip_clone: bool) -> str:
    """Clone (or reuse) the PDNC repo; checkout the pinned SHA; return HEAD."""
    if not (pdnc_dir / ".git").exists():
        if skip_clone:
            raise FileNotFoundError(f"--skip-pdnc-clone set but no checkout at {pdnc_dir}")
        pdnc_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", common.PDNC_REPO_URL, str(pdnc_dir)],
            check=True,
            timeout=1200,
        )
    if sha:
        try:
            subprocess.run(
                ["git", "-C", str(pdnc_dir), "checkout", "--quiet", sha], check=True, timeout=300
            )
        except subprocess.CalledProcessError:
            # A pre-existing checkout may predate a pin bump — fetch, then retry
            # (fail loud if the SHA is genuinely absent upstream).
            subprocess.run(
                ["git", "-C", str(pdnc_dir), "fetch", "--quiet", "origin"],
                check=True,
                timeout=1200,
            )
            subprocess.run(
                ["git", "-C", str(pdnc_dir), "checkout", "--quiet", sha], check=True, timeout=300
            )
    head = subprocess.run(
        ["git", "-C", str(pdnc_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    print(f"[i931-p0] PDNC checkout at {head}")
    return head


def _find_file(novel_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = novel_dir / name
        if p.exists():
            return p
    return None


def discover_novels(pdnc_dir: Path) -> list[Path]:
    """Novel dirs = subdirs of <root>/data (fallback: root) with text + CSVs."""
    roots = [pdnc_dir / "data", pdnc_dir]
    novels: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (
                _find_file(d, TEXT_CANDIDATES)
                and _find_file(d, QUOTE_CANDIDATES)
                and _find_file(d, CHAR_CANDIDATES)
            ):
                novels.append(d)
        if novels:
            break
    assert novels, f"no PDNC novel dirs with text+quotations+characters under {pdnc_dir}"
    return novels


def _parse_listish(val: str) -> list:
    """Parse a CSV cell holding a python/json list repr; [] on empty."""
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        out = ast.literal_eval(s)
        return list(out) if isinstance(out, (list, tuple)) else [out]
    except (ValueError, SyntaxError):
        return [s]


def _col(df, candidates: tuple[str, ...], path: Path) -> str:
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in cols:
            return cols[key]
    raise KeyError(f"{path}: none of columns {candidates} present (have {list(df.columns)})")


def load_novel(novel_dir: Path) -> dict:
    """Load one novel: text, quotations (speaker + spans), character aliases."""
    import pandas as pd

    text_path = _find_file(novel_dir, TEXT_CANDIDATES)
    quote_path = _find_file(novel_dir, QUOTE_CANDIDATES)
    char_path = _find_file(novel_dir, CHAR_CANDIDATES)
    text = text_path.read_text(encoding="utf-8")

    qdf = pd.read_csv(quote_path)
    span_col = _col(qdf, ("quoteByteSpans", "byteSpans", "quoteSpans", "spans"), quote_path)
    speaker_col = _col(qdf, ("speaker", "speakerName"), quote_path)
    qtext_col = _col(qdf, ("quoteText", "text", "quote"), quote_path)

    cdf = pd.read_csv(char_path)
    name_col = _col(cdf, ("Main Name", "mainName", "name", "character"), char_path)
    import contextlib

    alias_col = None
    with contextlib.suppress(KeyError):
        alias_col = _col(cdf, ("Aliases", "alias", "aliasList"), char_path)

    characters: dict[str, list[str]] = {}
    for _, row in cdf.iterrows():
        main = str(row[name_col]).strip()
        if not main or main.lower() == "nan":
            continue
        aliases = {main}
        if alias_col is not None:
            aliases.update(str(a).strip() for a in _parse_listish(row[alias_col]) if str(a).strip())
        characters[main] = sorted(aliases)

    quotes = []
    n_span_unparsed = 0
    for _, row in qdf.iterrows():
        speaker = str(row[speaker_col]).strip()
        spans = _parse_listish(row[span_col])
        # Normalize to a list of [start, end] pairs (a single flat pair is fine).
        if len(spans) == 2 and all(isinstance(v, (int, float)) for v in spans):
            spans = [spans]
        pair_spans = []
        for sp in spans:
            if isinstance(sp, (list, tuple)) and len(sp) == 2:
                pair_spans.append((int(sp[0]), int(sp[1])))
        if not pair_spans:
            # Counted, never silent (r1 Minor): systematic drift is caught by the
            # span-interpretation assert + mismatch gate; this reports the residue.
            n_span_unparsed += 1
            continue
        quotes.append({"speaker": speaker, "spans": pair_spans, "quote_text": str(row[qtext_col])})
    assert quotes, f"{novel_dir.name}: zero parseable quotations"
    if n_span_unparsed:
        print(f"[i931-p0] {novel_dir.name}: {n_span_unparsed} quote rows with unparseable spans")
    return {
        "novel_id": novel_dir.name,
        "text": text,
        "quotes": quotes,
        "characters": characters,
        "n_quotes_span_unparsed": n_span_unparsed,
    }


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def resolve_span_interpretation(novel: dict) -> tuple[str, float]:
    """Decide whether PDNC spans index CHARS or BYTES of the novel text.

    Samples up to 50 quotations, compares text[qs:qe] under each interpretation
    against the quoteText column (whitespace-normalized containment either
    way). Fails loud when neither interpretation matches >= 60% — a format
    drift we must inspect, never silently mis-slice.
    """
    text = novel["text"]
    tbytes = text.encode("utf-8")
    sample = novel["quotes"][:: max(1, len(novel["quotes"]) // 50)][:50]

    def _match_rate(mode: str) -> float:
        ok = 0
        for q in sample:
            qs, qe = q["spans"][0]
            if mode == "char":
                got = text[qs:qe] if qe <= len(text) else ""
            else:
                got = tbytes[qs:qe].decode("utf-8", errors="ignore") if qe <= len(tbytes) else ""
            got_n = _normalize_ws(got)
            want_n = _normalize_ws(q["quote_text"])[:80]
            if got_n and want_n and (want_n[:40] in got_n or got_n[:40] in want_n):
                ok += 1
        return ok / max(1, len(sample))

    char_rate, byte_rate = _match_rate("char"), _match_rate("byte")
    mode = "char" if char_rate >= byte_rate else "byte"
    rate = max(char_rate, byte_rate)
    assert rate >= 0.6, (
        f"{novel['novel_id']}: span interpretation unresolved "
        f"(char={char_rate:.2f}, byte={byte_rate:.2f}) — PDNC format drift, inspect"
    )
    return mode, rate


def byte_to_char_map(text: str) -> np.ndarray:
    """char_byte_starts[i] = byte offset of char i (for byte->char searchsorted)."""
    lens = np.fromiter((len(ch.encode("utf-8")) for ch in text), dtype=np.int64, count=len(text))
    starts = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(lens, out=starts[1:])
    return starts


# ---------------------------------------------------------------------------
# Arm A pair construction
# ---------------------------------------------------------------------------


def _alias_first_mention(text: str, aliases: list[str], lo: int, hi: int) -> int | None:
    """Earliest word-boundary alias occurrence in text[lo:hi); None if absent."""
    window = text[lo:hi]
    best = None
    for alias in aliases:
        if len(alias) < 2:
            continue
        m = re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", window)
        if m and (best is None or m.start() < best):
            best = m.start()
    return None if best is None else lo + best


def build_arma_pairs(novel: dict, tokenizer, *, window_tokens: int) -> dict:  # noqa: C901 -- linear per-novel QC + pair loop
    """All (window, character) pairs for one novel + per-novel QC counters."""
    text = novel["text"]
    novel_id = novel["novel_id"]
    ids, offsets = common.tokenize_with_offsets(tokenizer, text)
    bounds = common.sentence_bounds(text)
    span_mode, span_match_rate = resolve_span_interpretation(novel)
    b2c = byte_to_char_map(text) if span_mode == "byte" else None

    def to_char(pos: int) -> int:
        if span_mode == "char":
            return min(pos, len(text))
        return int(np.searchsorted(b2c, min(pos, int(b2c[-1])), side="right") - 1)

    # Character-name resolution (exclude groups / unknowable / narrator).
    valid_chars = {
        name: aliases
        for name, aliases in novel["characters"].items()
        if not NON_CHARACTER_SPEAKER_RE.match(name)
    }

    counters = {
        "quotes_total": 0,
        "quotes_speaker_excluded": 0,
        "quotes_span_mismatch": 0,
        "quotes_window_straddle": 0,
        "quotes_kept": 0,
    }
    # Per (character) quote token spans: (cov_lo, cov_hi, in_lo, in_hi).
    char_quotes: dict[str, list[tuple[int, int, int, int]]] = {}
    for q in novel["quotes"]:
        counters["quotes_total"] += 1
        speaker = q["speaker"]
        if speaker not in valid_chars:
            counters["quotes_speaker_excluded"] += 1
            continue
        mismatch = False
        segs = []
        for qs_raw, qe_raw in q["spans"]:
            qs, qe = to_char(qs_raw), to_char(qe_raw)
            if not (0 <= qs < qe <= len(text)):
                mismatch = True
                break
            cov_lo, cov_hi = common.covering_token_span(offsets, qs, qe)
            cs, ce = common.strip_quote_delims(text, qs, qe)
            in_lo, in_hi = common.inner_token_span(offsets, cs, ce) if cs < ce else (0, 0)
            # Edge check: covering span overhangs the exact boundaries by > 2
            # tokens on either edge -> alignment mismatch, drop (plan 4.0).
            if in_lo >= in_hi or (in_lo - cov_lo) > 2 or (cov_hi - in_hi) > 2:
                mismatch = True
                break
            segs.append((cov_lo, cov_hi, in_lo, in_hi))
        if mismatch or not segs:
            counters["quotes_span_mismatch"] += 1
            continue
        counters["quotes_kept"] += 1
        char_quotes.setdefault(speaker, []).extend(segs)
    for segs in char_quotes.values():
        segs.sort(key=lambda s: s[0])

    n_windows = len(ids) // window_tokens + (1 if len(ids) % window_tokens else 0)
    windows: list[dict] = []
    pairs: list[common.PairSpec] = []
    for w in range(n_windows):
        w_lo, w_hi = w * window_tokens, min((w + 1) * window_tokens, len(ids))
        if w_hi - w_lo < common.INTRO_MIN_TOKENS + common.TARGET_MIN_TOKENS:
            continue
        window_id = f"{novel_id}:w{w:04d}"
        w_char_lo = int(offsets[w_lo, 0])
        w_char_hi = int(offsets[w_hi - 1, 1])
        window_pairs: list[common.PairSpec] = []
        for speaker, segs in char_quotes.items():
            # Quotations fully inside the window (straddlers dropped + counted).
            in_win = [s for s in segs if s[0] >= w_lo and s[1] <= w_hi]
            n_straddle = sum(1 for s in segs if (s[0] < w_hi and s[1] > w_lo) and s not in in_win)
            counters["quotes_window_straddle"] += n_straddle
            if not in_win:
                continue
            mention = _alias_first_mention(text, valid_chars[speaker], w_char_lo, w_char_hi)
            if mention is None:
                continue
            built = common.build_intro_and_targets(
                text=text,
                offsets=offsets,
                excerpt_tok=(w_lo, w_hi),
                mention_char=mention,
                quote_spans_tok=in_win,
                bounds=bounds,
            )
            if built is None:
                continue
            (c_s, c_e), t_spans = built
            t_min = min(lo for lo, _ in t_spans)
            pair = common.PairSpec(
                row_id=f"{window_id}:{speaker}",
                group_id=novel_id,
                char_id=speaker,
                c_span=(c_s - w_lo, c_e - w_lo),
                t_spans=[(lo - w_lo, hi - w_lo) for lo, hi in t_spans],
                ctx_span=(0, t_min - w_lo),
                meta={
                    "window_id": window_id,
                    "c_text": text[int(offsets[c_s, 0]) : int(offsets[c_e - 1, 1])],
                    "n_t_tokens": int(sum(hi - lo for lo, hi in t_spans)),
                },
            )
            pair.validate(w_hi - w_lo)
            window_pairs.append(pair)
        if window_pairs:
            windows.append(
                {
                    "window_id": window_id,
                    "novel_id": novel_id,
                    "window_idx": w,
                    "input_ids": ids[w_lo:w_hi],
                }
            )
            pairs.extend(window_pairs)

    considered = counters["quotes_total"] - counters["quotes_speaker_excluded"]
    mismatch_rate = counters["quotes_span_mismatch"] / max(1, considered)
    return {
        "novel_id": novel_id,
        "windows": windows,
        "pairs": pairs,
        "counters": counters,
        "span_mode": span_mode,
        "span_match_rate": span_match_rate,
        "mismatch_rate": mismatch_rate,
        "n_tokens": len(ids),
    }


# ---------------------------------------------------------------------------
# Arm C (WikiText-103-raw separator anchors)
# ---------------------------------------------------------------------------

_WIKI_HEADER_RE = re.compile(r"^ ?= [^=].* = ?$")


def iter_wikitext_articles(max_articles: int):
    """Yield (title, text) articles assembled from the streamed train split."""
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
        revision=common.WIKITEXT_REVISION,  # r2 fix: pin the HF dataset revision
    )
    title, buf = None, []
    n = 0
    for row in ds:
        line = row["text"]
        if _WIKI_HEADER_RE.match(line.rstrip("\n")):
            if title is not None and buf:
                yield title, "".join(buf)
                n += 1
                if n >= max_articles:
                    return
            title, buf = line.strip().strip("= ").strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None and buf and n < max_articles:
        yield title, "".join(buf)


def build_armc_pairs(  # noqa: C901 -- linear anchor-eligibility ladder
    tokenizer, *, n_articles: int, max_anchors: int, pool_multiplier: int = 3
) -> dict:
    """Separator-anchor + preceding-sentence pairs from WikiText-103-raw."""
    pool: list[dict] = []
    target_pool = n_articles * pool_multiplier
    for k, (title, text) in enumerate(iter_wikitext_articles(target_pool * 2)):
        ids, offsets = common.tokenize_with_offsets(tokenizer, text)
        if len(ids) < common.ARMC_ARTICLE_MIN_TOKENS:
            continue
        cap = min(len(ids), common.ARMC_ARTICLE_CAP_TOKENS)
        pool.append(
            {
                "article_idx": k,
                "title": title,
                "text": text[: int(offsets[cap - 1, 1])],
                "ids": ids[:cap],
                "offsets": offsets[:cap],
            }
        )
        if len(pool) >= target_pool:
            break
    assert pool, "no WikiText articles collected"
    rng = np.random.default_rng(common.BUILD_SEED)
    take = rng.choice(len(pool), size=min(n_articles, len(pool)), replace=False)
    articles_out, pairs_out = [], []
    for ai in sorted(int(v) for v in take):
        art = pool[ai]
        ids, offsets, text = art["ids"], art["offsets"], art["text"]
        bounds = common.sentence_bounds(text)
        # Sentence-final anchor tokens: the token containing the sentence's
        # final [.!?] whose own text strips to exactly that punctuation.
        anchors = []
        for s, e in bounds:
            seg = text[s:e].rstrip()
            if not seg or seg[-1] not in ".!?":
                continue
            pi = s + len(seg) - 1
            lo, hi = common.covering_token_span(offsets, pi, pi + 1)
            if hi - lo != 1:
                continue
            t = lo
            tok_text = text[int(offsets[t, 0]) : int(offsets[t, 1])].strip()
            if tok_text in (".", "!", "?"):
                anchors.append((t, s))
        if len(anchors) < 2:
            continue
        anchor_positions = [a for a, _ in anchors]
        eligible = []
        for j, (t, sent_start) in enumerate(anchors):
            nxt = anchor_positions[j + 1] if j + 1 < len(anchors) else len(ids)
            span_lo, span_hi = t + 1, nxt
            if not (common.ARMC_SPAN_MIN <= span_hi - span_lo <= common.ARMC_SPAN_MAX):
                continue
            ps_lo, ps_hi = common.inner_token_span(offsets, sent_start, int(offsets[t, 0]))
            ps_hi = min(ps_hi, t)
            ps_lo = max(ps_lo, ps_hi - common.ARMC_PREV_CAP_TOKENS)  # keep LAST <=96 tokens
            if ps_hi - ps_lo < common.ARMC_PREV_MIN_TOKENS:
                continue
            eligible.append((t, span_lo, span_hi, ps_lo, ps_hi))
        if not eligible:
            continue
        art_rng = np.random.default_rng(common.BUILD_SEED + art["article_idx"])
        keep = art_rng.choice(len(eligible), size=min(max_anchors, len(eligible)), replace=False)
        article_id = f"wiki:{art['article_idx']:05d}"
        art_pairs = []
        for j in sorted(int(v) for v in keep):
            t, span_lo, span_hi, ps_lo, ps_hi = eligible[j]
            pair = common.PairSpec(
                row_id=f"{article_id}:a{t}",
                group_id=article_id,
                char_id="sep",
                c_span=(ps_lo, ps_hi),
                t_spans=[(span_lo, span_hi)],
                ctx_span=(ps_lo, ps_hi),
                meta={"window_id": article_id, "anchor_pos": int(t)},
            )
            pair.validate(len(ids), min_c=common.ARMC_PREV_MIN_TOKENS, min_t=common.ARMC_SPAN_MIN)
            assert pair.meta["anchor_pos"] < span_lo, "anchor must precede its span"
            art_pairs.append(pair)
        if art_pairs:
            articles_out.append(
                {
                    "window_id": article_id,
                    "novel_id": article_id,
                    "window_idx": 0,
                    "title": art["title"],
                    "input_ids": ids,
                }
            )
            pairs_out.extend(art_pairs)
    return {"articles": articles_out, "pairs": pairs_out}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"[i931-p0] wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    pairs_dir = args.data_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = common.get_tokenizer()

    print("[phase=p0_pairs] building Arm-A pairs")
    pdnc_sha = stage_pdnc(args.pdnc_dir, args.pdnc_sha, skip_clone=args.skip_pdnc_clone)
    novel_dirs = discover_novels(args.pdnc_dir)
    if args.max_novels:
        novel_dirs = novel_dirs[: args.max_novels]
    print(f"[i931-p0] {len(novel_dirs)} novels: {[d.name for d in novel_dirs]}")

    per_novel = []
    all_windows: list[dict] = []
    all_pairs: list[common.PairSpec] = []
    dropped_novels = []
    for d in novel_dirs:
        novel = load_novel(d)
        res = build_arma_pairs(novel, tokenizer, window_tokens=args.window_tokens)
        row = {
            "novel_id": res["novel_id"],
            "span_mode": res["span_mode"],
            "span_match_rate": res["span_match_rate"],
            "mismatch_rate": res["mismatch_rate"],
            "n_pairs": len(res["pairs"]),
            "n_windows": len(res["windows"]),
            "n_tokens": res["n_tokens"],
            "n_quotes_span_unparsed": novel.get("n_quotes_span_unparsed", 0),
            **res["counters"],
        }
        per_novel.append(row)
        if res["mismatch_rate"] > 0.10:
            dropped_novels.append(res["novel_id"])
            print(
                f"[i931-p0] G0 DROP novel {res['novel_id']}: mismatch "
                f"{res['mismatch_rate']:.3f} > 0.10"
            )
            continue
        all_windows.extend(res["windows"])
        all_pairs.extend(res["pairs"])
        print(
            f"[i931-p0] {res['novel_id']}: {len(res['pairs'])} pairs / "
            f"{len(res['windows'])} windows (mismatch {res['mismatch_rate']:.3f})"
        )

    # Seeded novel-stratified subsample to the cap.
    subsampled_from = len(all_pairs)
    if len(all_pairs) > args.max_pairs:
        gids = np.asarray([p.group_id for p in all_pairs])
        keep = common.group_stratified_subsample(gids, args.max_pairs, seed=common.BUILD_SEED)
        keep_set = set(int(v) for v in keep)
        all_pairs = [p for i, p in enumerate(all_pairs) if i in keep_set]
        kept_windows = {p.meta["window_id"] for p in all_pairs}
        all_windows = [w for w in all_windows if w["window_id"] in kept_windows]
        print(f"[i931-p0] subsampled pairs {subsampled_from} -> {len(all_pairs)}")

    novels_kept = sorted({p.group_id for p in all_pairs})
    g0 = {
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(all_pairs)),
        "pdnc_sha": pdnc_sha,
        "per_novel": per_novel,
        "dropped_novels": dropped_novels,
        "n_pairs": len(all_pairs),
        "n_pairs_before_subsample": subsampled_from,
        "n_novels_kept": len(novels_kept),
        "n_windows": len(all_windows),
        "gate_min_pairs": 800,
        "gate_min_novels": 10,
        "smoke": bool(args.smoke),
        "pass": bool(len(all_pairs) >= 800 and len(novels_kept) >= 10),
    }
    common.write_json(args.out_dir / "g0_gate.json", g0)

    _write_jsonl(pairs_dir / "windows_armA.jsonl", all_windows)
    _write_jsonl(pairs_dir / "pairs_armA.jsonl", [p.to_dict() for p in all_pairs])

    print("[phase=p0_battery] building Arm-B prompt battery")
    battery = common.build_prompt_battery(seed=common.BUILD_SEED)
    if args.smoke:
        battery = battery[:20]
    common.write_json(
        pairs_dir / "prompt_battery.json",
        {
            "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(battery)),
            "prompts": battery,
        },
    )

    armc_counts = {"articles": 0, "pairs": 0}
    if not args.skip_armc:
        print("[phase=p0_armc] building Arm-C separator pairs")
        armc = build_armc_pairs(tokenizer, n_articles=args.n_articles, max_anchors=args.max_anchors)
        _write_jsonl(pairs_dir / "articles_armC.jsonl", armc["articles"])
        _write_jsonl(pairs_dir / "pairs_armC.jsonl", [p.to_dict() for p in armc["pairs"]])
        armc_counts = {"articles": len(armc["articles"]), "pairs": len(armc["pairs"])}

    meta = {
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(all_pairs)),
        "pdnc_sha": pdnc_sha,
        "pdnc_sha_pin": common.PDNC_SHA,
        "wikitext_revision": common.WIKITEXT_REVISION,
        "tokenizer": common.MODEL_ID,
        "window_tokens": args.window_tokens,
        "arma": {"n_pairs": len(all_pairs), "n_windows": len(all_windows)},
        "armc": armc_counts,
        "per_novel": per_novel,
        "dropped_novels": dropped_novels,
        # Eyeball sample (assumption 17): 20 realized intro spans with texts.
        "intro_span_sample": [
            {"row_id": p.row_id, "c_span": list(p.c_span), "c_text": p.meta.get("c_text", "")}
            for p in all_pairs[:20]
        ],
    }
    common.write_json(pairs_dir / "pairs_meta.json", meta)

    if not args.smoke and not g0["pass"]:
        print("[i931-p0] G0 FAIL — see g0_gate.json", file=sys.stderr)
        return 3
    print("[i931-p0] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
