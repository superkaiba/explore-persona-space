#!/usr/bin/env python3
"""Issue #1489 free-analysis follow-up: judge-independent lexical manipulation screen.

For each of the 16 augmentation instances, compares a deterministic per-row
"cue hit" between the augmented cell's generations and the plain cell's
generations on the SAME base rows (paired join on ``base_row_id``):

- Instances with a deterministic code validator (``CODE_VALIDATORS`` — the 4
  format instances + ``instr_concise``) use the validator as the hit function
  (that judge-independent axis already exists; included for table completeness).
- All other instances use a MECHANICALLY derived cue set from the instance's
  own augmentation text (``build_augmentation_library()``): quoted phrases
  (>= 3 chars) matched as case-insensitive substrings, plus content words
  (>= 4 chars, standard-stopword-filtered, possessive-stripped) matched as
  word-boundary stems (trailing "s" stripped for words >= 5 chars). Words that
  appear in >= ``BOILERPLATE_DF`` of the 16 augmentation texts are excluded as
  library boilerplate ("user", "response", "answer", ...) — a reproducible,
  data-independent genericity filter (no hand curation).

Statistics per instance (n = paired rows, one pair per shared base row):
paired delta (aug rate - plain rate), 95% bootstrap CI via a seeded
multinomial resample of the discordant-pair categories (equivalent to
pair-resampling for this statistic), exact McNemar p (binomial test on
discordant pairs), and Clopper-Pearson 95% CIs per arm. A relevance-stratum
breakdown is added where the conditions manifest defines one (scoped facts +
instr_refuse_code).

CPU-only, no model calls, deterministic (fixed bootstrap seed). Never prints
completion text (counts/rates only). Output:
``eval_results/issue_1489/lexical_screen.json``.

Run:
    uv run python scripts/issue1489_lexical_screen.py
    uv run python scripts/issue1489_lexical_screen.py --self-test
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before numpy/scipy so the shared-VM BLAS thread-cap setdefaults (#847) bind
# in-process even on a bare (prefix-less) launch.
load_dotenv()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue1489_common import (  # noqa: E402
    AUGMENT_SLUGS,
    CODE_VALIDATORS,
    build_augmentation_library,
    cell_for_slug,
    load_conditions_manifest,
    rows_for_cell,
)

BOOTSTRAP_SEED = 20260718
N_BOOTSTRAP = 10_000
MIN_WORD_LEN = 4
MIN_PHRASE_LEN = 3
# A content word appearing in >= this many of the 16 augmentation texts is
# library boilerplate, not an instance-specific cue (mechanical genericity filter).
BOILERPLATE_DF = 3

# Standard English function words (fixed, embedded for determinism). Deliberately
# kept to function words + bare fillers — instance-specific boilerplate removal
# is the job of the BOILERPLATE_DF cross-instance filter, not this list.
STOPWORDS = frozenset(
    """
    a about above after again against all also an and any are aren as at be been
    before being below between both but by can cannot could did do does doing don
    down during each even ever few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just like may me
    might more most must my myself never no nor not now of off on once only onto or
    other our ours ourselves out over own per please same shall she should so some
    still such than that the their theirs them themselves then there these they
    this those through to too under until up upon us very via was we well were what
    when where which while who whom why will with within without would you your
    yours yourself yourselves
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*|\d{4}")
_QUOTED_RE = re.compile(rf"'([^']{{{MIN_PHRASE_LEN},60}})'|\"([^\"]{{{MIN_PHRASE_LEN},60}})\"")


def _content_words(text: str) -> list[str]:
    """Ordered, deduped content words of an augmentation text (pre-df-filter)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _WORD_RE.findall(text):
        w = raw.lower().strip("'")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) < MIN_WORD_LEN or w in STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _quoted_phrases(text: str) -> list[str]:
    """Ordered, deduped quoted spans (>= MIN_PHRASE_LEN chars) of an augmentation text."""
    out: list[str] = []
    seen: set[str] = set()
    for g1, g2 in _QUOTED_RE.findall(text):
        p = (g1 or g2).strip()
        key = p.lower()
        if len(p) >= MIN_PHRASE_LEN and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def derive_cue_sets(library: dict[str, dict]) -> dict[str, dict]:
    """slug -> {words, phrases} for every NON-validator instance (fail-loud on empty).

    The boilerplate document-frequency filter is computed over ALL 16 texts
    (validator instances included) so it is a property of the library alone.
    """
    per_slug_words = {slug: _content_words(meta["text"]) for slug, meta in library.items()}
    df: dict[str, int] = {}
    for words in per_slug_words.values():
        for w in set(words):
            df[w] = df.get(w, 0) + 1
    boilerplate = {w for w, n in df.items() if n >= BOILERPLATE_DF}

    cues: dict[str, dict] = {}
    for slug, meta in library.items():
        if slug in CODE_VALIDATORS:
            continue
        words = [w for w in per_slug_words[slug] if w not in boilerplate]
        phrases = _quoted_phrases(meta["text"])
        if not words and not phrases:
            raise RuntimeError(
                f"empty cue set for {slug!r} after stopword/boilerplate filtering — "
                "the mechanical derivation cannot screen this instance"
            )
        cues[slug] = {
            "words": words,
            "phrases": phrases,
            "boilerplate_excluded": sorted(w for w in per_slug_words[slug] if w in boilerplate),
        }
    return cues


def _word_pattern(word: str) -> str:
    """Word-boundary stem pattern: trailing 's' stripped for words >= 5 chars."""
    stem = word[:-1] if word.endswith("s") and len(word) >= 5 else word
    return r"\b" + re.escape(stem) + r"\w*"


def _phrase_pattern(phrase: str) -> str:
    """Whitespace-normalized case-insensitive substring pattern for a quoted phrase."""
    parts = [re.escape(tok) for tok in phrase.split()]
    return r"\s+".join(parts)


def compile_cue_regexes(words: list[str], phrases: list[str]) -> tuple[re.Pattern, dict]:
    """(combined any-cue regex, per-cue compiled regexes keyed by display name)."""
    per_cue: dict[str, re.Pattern] = {}
    pats: list[str] = []
    for p in phrases:
        pat = _phrase_pattern(p)
        per_cue[f"phrase:{p}"] = re.compile(pat, re.IGNORECASE)
        pats.append(pat)
    for w in words:
        pat = _word_pattern(w)
        per_cue[f"word:{w}"] = re.compile(pat, re.IGNORECASE)
        pats.append(pat)
    return re.compile("|".join(pats), re.IGNORECASE), per_cue


def load_cell_completions(gen_dir: Path, cell_id: str) -> dict[str, str]:
    """base_row_id -> completion for one cell, fail-loud on shape violations."""
    shards = sorted((gen_dir / cell_id).glob("shard*.json"))
    if not shards:
        raise RuntimeError(f"no generation shards found under {gen_dir / cell_id}")
    out: dict[str, str] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        for row in payload["rows"]:
            rid = row["base_row_id"]
            comp = row["completion"]
            if not isinstance(comp, str):
                raise RuntimeError(f"non-str completion in {shard} (base_row_id={rid})")
            if rid in out:
                raise RuntimeError(f"duplicate base_row_id {rid} in cell {cell_id}")
            out[rid] = comp
    return out


def join_paired(
    aug_map: dict[str, str], plain_map: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """(base_row_ids, aug_completions, plain_completions), fail-loud on join gaps."""
    if not aug_map:
        raise RuntimeError("augmented cell has zero rows")
    missing = [rid for rid in aug_map if rid not in plain_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} augmented base_row_ids missing from cell_plain (first: {missing[0]!r})"
        )
    ids = sorted(aug_map)  # deterministic order
    return ids, [aug_map[rid] for rid in ids], [plain_map[rid] for rid in ids]


def paired_stats(aug_hits: np.ndarray, plain_hits: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired-rate stats: rates, per-arm exact CIs, paired delta + bootstrap CI, McNemar p.

    The bootstrap resamples the per-pair difference categories (+1/-1/0) via a
    multinomial draw — equivalent to resampling pairs for the delta statistic.
    """
    n = int(aug_hits.size)
    if n == 0 or aug_hits.size != plain_hits.size:
        raise RuntimeError(f"bad paired arrays: n_aug={aug_hits.size} n_plain={plain_hits.size}")
    aug_k = int(aug_hits.sum())
    plain_k = int(plain_hits.sum())
    b = int(np.sum(aug_hits & ~plain_hits))  # aug-only hits
    c = int(np.sum(~aug_hits & plain_hits))  # plain-only hits
    delta = (b - c) / n

    probs = np.array([b / n, c / n, (n - b - c) / n])
    draws = rng.multinomial(n, probs, size=N_BOOTSTRAP)  # (N_BOOTSTRAP, 3)
    deltas = (draws[:, 0] - draws[:, 1]) / n
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    mcnemar_p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)

    def cp_ci(k: int) -> list[float]:
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        return [float(ci.low), float(ci.high)]

    return {
        "n_pairs": n,
        "aug_hits": aug_k,
        "plain_hits": plain_k,
        "aug_rate": aug_k / n,
        "plain_rate": plain_k / n,
        "aug_rate_ci95": cp_ci(aug_k),
        "plain_rate_ci95": cp_ci(plain_k),
        "paired_delta": delta,
        "delta_ci95": [float(ci_lo), float(ci_hi)],
        "discordant": {"aug_only": b, "plain_only": c},
        "mcnemar_exact_p": mcnemar_p,
    }


def git_sha() -> str:
    """Current worktree HEAD (reproducibility metadata)."""
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        env={**os.environ},
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def run_self_test() -> None:
    """Degenerate-input probes: each data gate fires with its designed handling."""
    # Gate 1: paired-join completeness raises on a missing plain row.
    try:
        join_paired({"r1": "x", "r2": "y"}, {"r1": "x"})
        raise AssertionError("join gate did not fire")
    except RuntimeError as e:
        assert "missing from cell_plain" in str(e)
        print("[self-test] join-completeness gate: PASS (RuntimeError raised)")
    # Gate 2: empty cue set after filtering raises.
    fake_lib = {
        "fact_stoponly": {"family": "fact", "text": "the of and to a", "relevant_topics": None}
    }
    try:
        derive_cue_sets(fake_lib)
        raise AssertionError("empty-cue gate did not fire")
    except RuntimeError as e:
        assert "empty cue set" in str(e)
        print("[self-test] empty-cue-set gate: PASS (RuntimeError raised)")
    # Gate 3: degenerate stats (no discordant pairs) -> delta 0, CI [0,0], p=1.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    s = paired_stats(np.array([True, False]), np.array([True, False]), rng)
    assert s["paired_delta"] == 0.0 and s["mcnemar_exact_p"] == 1.0
    assert s["delta_ci95"] == [0.0, 0.0]
    print("[self-test] degenerate-discordant stats: PASS (delta=0, CI=[0,0], p=1)")
    # Gate 4: zero-length pairing raises.
    try:
        paired_stats(np.array([], dtype=bool), np.array([], dtype=bool), rng)
        raise AssertionError("empty-pairs gate did not fire")
    except RuntimeError as e:
        assert "bad paired arrays" in str(e)
        print("[self-test] empty-pairs gate: PASS (RuntimeError raised)")
    print("[self-test] all gates PASS")


def main() -> None:
    """Run the 16-instance paired lexical screen and write the output JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=REPO_ROOT / "data/issue_1489/p5_out/raw_completions/generation",
    )
    ap.add_argument("--conditions-dir", type=Path, default=REPO_ROOT / "data/issue_1489/conditions")
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "eval_results/issue_1489/lexical_screen.json"
    )
    ap.add_argument("--self-test", action="store_true", help="run degenerate-input gate probes")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    library = build_augmentation_library(repo_root=REPO_ROOT)
    cue_sets = derive_cue_sets(library)
    manifest = load_conditions_manifest(args.conditions_dir)

    plain_map = load_cell_completions(args.gen_dir, "cell_plain")
    print(f"[lexical-screen] cell_plain rows: {len(plain_map)}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    instances: list[dict] = []
    for slug in AUGMENT_SLUGS:
        cell_id = cell_for_slug(slug)
        aug_map = load_cell_completions(args.gen_dir, cell_id)
        ids, aug_comps, plain_comps = join_paired(aug_map, plain_map)

        # relevance per aug row (property of (augmentation, query); may be None)
        man_rows = rows_for_cell(manifest, cell_id)
        rel_by_base: dict[str, bool | None] = {}
        for r in man_rows:
            if r["base_row_id"] in rel_by_base:
                raise RuntimeError(f"duplicate base_row_id in manifest for {cell_id}")
            rel_by_base[r["base_row_id"]] = r.get("relevant")
        unmapped = [rid for rid in ids if rid not in rel_by_base]
        if unmapped:
            raise RuntimeError(f"{len(unmapped)} generation rows missing from manifest ({cell_id})")

        if slug in CODE_VALIDATORS:
            validator = CODE_VALIDATORS[slug]
            aug_hits = np.array([validator(t) for t in aug_comps], dtype=bool)
            plain_hits = np.array([validator(t) for t in plain_comps], dtype=bool)
            row: dict = {
                "slug": slug,
                "family": library[slug]["family"],
                "cell_id": cell_id,
                "cue_kind": "code_validator",
                "validator": validator.__name__,
                "cue_words": None,
                "cue_phrases": None,
            }
            per_cue = None
        else:
            cs = cue_sets[slug]
            combined, per_cue_re = compile_cue_regexes(cs["words"], cs["phrases"])
            aug_hits = np.array([bool(combined.search(t)) for t in aug_comps], dtype=bool)
            plain_hits = np.array([bool(combined.search(t)) for t in plain_comps], dtype=bool)
            per_cue = {
                name: {
                    "aug_rate": float(np.mean([bool(rx.search(t)) for t in aug_comps])),
                    "plain_rate": float(np.mean([bool(rx.search(t)) for t in plain_comps])),
                }
                for name, rx in per_cue_re.items()
            }
            row = {
                "slug": slug,
                "family": library[slug]["family"],
                "cell_id": cell_id,
                "cue_kind": "lexical_cues",
                "validator": None,
                "cue_words": cs["words"],
                "cue_phrases": cs["phrases"],
                "boilerplate_excluded": cs["boilerplate_excluded"],
            }

        row.update(paired_stats(aug_hits, plain_hits, rng))
        if per_cue is not None:
            row["per_cue_rates"] = per_cue

        # relevance-stratum breakdown where the manifest defines one
        rel_flags = [rel_by_base[rid] for rid in ids]
        if any(f is True for f in rel_flags) and any(f is False for f in rel_flags):
            by_rel = {}
            for label, want in (("relevant", True), ("irrelevant", False)):
                mask = np.array([f is want for f in rel_flags], dtype=bool)
                by_rel[label] = paired_stats(aug_hits[mask], plain_hits[mask], rng)
            row["by_relevance"] = by_rel
        else:
            row["by_relevance"] = None

        instances.append(row)
        print(
            f"[lexical-screen] {slug}: n={row['n_pairs']} aug={row['aug_rate']:.4f} "
            f"plain={row['plain_rate']:.4f} delta={row['paired_delta']:+.4f} "
            f"CI=[{row['delta_ci95'][0]:+.4f},{row['delta_ci95'][1]:+.4f}] "
            f"p={row['mcnemar_exact_p']:.2e} ({row['cue_kind']})"
        )

    out = {
        "issue": 1489,
        "analysis": "lexical_manipulation_screen",
        "description": (
            "Judge-independent lexical manipulation screen: per-instance paired "
            "cue-hit rates, augmented vs plain generations on shared base rows."
        ),
        "meta": {
            "git_sha": git_sha(),
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "min_word_len": MIN_WORD_LEN,
            "min_phrase_len": MIN_PHRASE_LEN,
            "boilerplate_df_threshold": BOILERPLATE_DF,
            "n_stopwords": len(STOPWORDS),
            "gen_dir": str(args.gen_dir),
            "conditions_dir": str(args.conditions_dir),
            "n_plain_rows": len(plain_map),
        },
        "instances": instances,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[lexical-screen] wrote {args.out} ({len(instances)} instances)")


if __name__ == "__main__":
    main()
