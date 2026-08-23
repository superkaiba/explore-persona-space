"""Prompt-script x response-script cross-tab of #2477 base-chat coherence.

Free-analysis follow-up (Step 9a-ter): which prompt SCRIPT-CLASSES drive the
CJK-script drift in the base-under-chat-template condition (#2477 published
body, "Language-intrusion audit").

Classification rule (script-classes, NOT languages — Unicode script ranges
cannot reliably distinguish languages within a script; a diacritic-free
Romance/Germanic prompt classifies "latin_english"):

  1. ``cjk`` — text contains ANY character in the published intrusion class
     (one definition throughout #2477, reused verbatim from
     scripts/issue2225_fu1_cjk_audit.py): Han U+4E00-9FFF + Ext-A U+3400-4DBF
     + Compatibility U+F900-FAFF + kana U+3040-30FF + Hangul U+AC00-D7AF.
     Any-char (not majority) so the cross-tab's cjk prompt class equals the
     published intrusion carve-out ("a CJK reply to a CJK prompt is not
     intrusion"), and the cjk response class equals the published any-CJK
     disclosure column.
  2. Otherwise count LETTERS (unicodedata category L*) by script bucket:
     Latin-ASCII [A-Za-z]; Latin-extended (diacritic-bearing: Latin-1
     Supplement letters, Latin Extended-A/B, Latin Extended Additional);
     Cyrillic U+0400-052F; every other letter -> "other".
     - zero letters -> ``no_letters``
     - a bucket family holding > 0.5 of letters is dominant:
       Latin dominant -> ``latin_other`` when extended-Latin letters >= 2 and
       >= 0.5% of Latin letters, else ``latin_english``;
       Cyrillic dominant -> ``cyrillic``; other dominant -> ``other_mixed``
     - no majority -> ``other_mixed``

Intrusion (published definition): response contains any CJK AND prompt class
is not ``cjk``. Coherent: per-item mean judge score >= 50
(coherence_verdict.json per_item_mean_scores). Fail-fast reconciliation
asserts pin this script's rule to the published anchors (6 CJK prompts;
chat 93 any-CJK / 89 intrusions; bare 9 / 3).

Content hygiene: prompts/responses are real-user corpus text (LMSYS/WildChat)
— this script never prints or persists any text field; only counts,
fractions, and the classification rule leave it.

Usage (from repo root):
  uv run python scripts/issue2477_lang_crosstab.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847)

# Published #2477 intrusion class, verbatim from scripts/issue2225_fu1_cjk_audit.py:
# [一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]
CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

LATIN_ASCII = re.compile(r"[A-Za-z]")
# Diacritic-bearing Latin letters: Latin-1 Supplement letters (excl. multiply/divide
# signs U+00D7/U+00F7), Latin Extended-A/B, Latin Extended Additional.
LATIN_EXT = re.compile(r"[À-ÖØ-öø-ɏḀ-ỿ]")
CYRILLIC = re.compile(r"[Ѐ-ԯ]")

# Disclosure-only sub-buckets for the other_mixed class (meta breakdown).
OTHER_BLOCKS = {
    "greek": re.compile(r"[Ͱ-Ͽἀ-῿]"),
    "arabic": re.compile(r"[؀-ۿݐ-ݿ]"),
    "hebrew": re.compile(r"[֐-׿]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "thai": re.compile(r"[฀-๿]"),
}

CLASSES = ["latin_english", "latin_other", "cyrillic", "cjk", "other_mixed", "no_letters"]
COHERENT_THRESHOLD = 50.0


def classify(text: str) -> str:
    """Assign a script-class per the module docstring rule."""
    if CJK.search(text):
        return "cjk"
    n_ascii = len(LATIN_ASCII.findall(text))
    n_ext = len(LATIN_EXT.findall(text))
    n_cyr = len(CYRILLIC.findall(text))
    n_letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    n_other = n_letters - n_ascii - n_ext - n_cyr
    if n_other < 0:
        raise RuntimeError(f"letter accounting bug: other={n_other} (ranges overlap?)")
    if n_letters == 0:
        return "no_letters"
    n_latin = n_ascii + n_ext
    if n_latin / n_letters > 0.5:
        if n_ext >= 2 and n_ext / n_latin >= 0.005:
            return "latin_other"
        return "latin_english"
    if n_cyr / n_letters > 0.5:
        return "cyrillic"
    return "other_mixed"


def other_block_label(text: str) -> str:
    """Disclosure label for an other_mixed text: the largest known non-Latin block."""
    counts = {name: len(rx.findall(text)) for name, rx in OTHER_BLOCKS.items()}
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    return best if counts[best] > 0 else "unrecognized"


def cjk_letter_frac(text: str) -> float:
    n_letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    if n_letters == 0:
        return 0.0
    return len(CJK.findall(text)) / n_letters


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    if len(rows) != 200:
        raise RuntimeError(f"{path}: expected 200 rows, got {len(rows)}")
    return rows


def item_scores(verdict: dict, arm: str, prompt_idxs: list[int]) -> dict[int, float]:
    per_arm = verdict["per_item_mean_scores"][arm]
    scores: dict[int, float] = {}
    for idx in prompt_idxs:
        key = f"{arm}--{idx}"
        if key not in per_arm:
            raise RuntimeError(f"coherence_verdict.json missing per-item score {key}")
        scores[idx] = float(per_arm[key])
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-root", type=Path, default=Path("eval_results/issue_2477"))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default: <eval-root>/lang_crosstab/lang_crosstab.json)",
    )
    args = ap.parse_args()

    root: Path = args.eval_root
    out_path: Path = args.out or root / "lang_crosstab" / "lang_crosstab.json"
    chat_rows = load_jsonl(root / "fresh_completions" / "base_chat_seed42.jsonl")
    bare_rows = load_jsonl(root / "fresh_completions" / "base_bare_seed42.jsonl")
    verdict = json.loads((root / "coherence_verdict.json").read_text())

    chat_by_idx = {r["prompt_idx"]: r for r in chat_rows}
    bare_by_idx = {r["prompt_idx"]: r for r in bare_rows}
    if set(chat_by_idx) != set(bare_by_idx) or len(chat_by_idx) != 200:
        raise RuntimeError("prompt_idx sets differ between base_chat and base_bare (or dupes)")
    idxs = sorted(chat_by_idx)

    chat_scores = item_scores(verdict, "arm_base_chat", idxs)
    bare_scores = item_scores(verdict, "arm_base_bare", idxs)

    # Per-row classification (prompt text identical across arms by construction).
    for idx in idxs:
        if chat_by_idx[idx]["prompt"] != bare_by_idx[idx]["prompt"]:
            raise RuntimeError(f"prompt text differs across arms at prompt_idx={idx}")
    prompt_class = {idx: classify(chat_by_idx[idx]["prompt"]) for idx in idxs}
    chat_resp_class = {idx: classify(chat_by_idx[idx]["response"]) for idx in idxs}
    bare_resp_class = {idx: classify(bare_by_idx[idx]["response"]) for idx in idxs}

    def resp_is_cjk(rows_by_idx: dict, idx: int) -> bool:
        return bool(CJK.search(rows_by_idx[idx]["response"]))

    # Fail-fast reconciliation against the published #2477 anchors — pins this
    # script's CJK rule to the body's "one definition throughout".
    anchors = {
        "cjk_prompts": sum(1 for i in idxs if prompt_class[i] == "cjk"),
        "chat_any_cjk_responses": sum(1 for i in idxs if resp_is_cjk(chat_by_idx, i)),
        "chat_intrusions": sum(
            1 for i in idxs if resp_is_cjk(chat_by_idx, i) and prompt_class[i] != "cjk"
        ),
        "bare_any_cjk_responses": sum(1 for i in idxs if resp_is_cjk(bare_by_idx, i)),
        "bare_intrusions": sum(
            1 for i in idxs if resp_is_cjk(bare_by_idx, i) and prompt_class[i] != "cjk"
        ),
    }
    published = {
        "cjk_prompts": 6,
        "chat_any_cjk_responses": 93,
        "chat_intrusions": 89,
        "bare_any_cjk_responses": 9,
        "bare_intrusions": 3,
    }
    if anchors != published:
        raise RuntimeError(f"reconciliation FAILED: computed {anchors} vs published {published}")

    # Per prompt-class summary.
    summary: dict[str, dict] = {}
    for cls in CLASSES:
        rows = [i for i in idxs if prompt_class[i] == cls]
        if not rows:
            continue
        n = len(rows)
        chat_coh = sum(1 for i in rows if chat_scores[i] >= COHERENT_THRESHOLD)
        bare_coh = sum(1 for i in rows if bare_scores[i] >= COHERENT_THRESHOLD)
        chat_intr = sum(1 for i in rows if resp_is_cjk(chat_by_idx, i) and cls != "cjk")
        bare_intr = sum(1 for i in rows if resp_is_cjk(bare_by_idx, i) and cls != "cjk")
        # Among chat intrusions in this class: how many are majority-CJK responses
        # (>= 0.5 of letters), vs. a stray character (disclosure).
        chat_intr_majority = sum(
            1
            for i in rows
            if resp_is_cjk(chat_by_idx, i)
            and cls != "cjk"
            and cjk_letter_frac(chat_by_idx[i]["response"]) >= 0.5
        )
        summary[cls] = {
            "n": n,
            "chat_frac_coherent": round(chat_coh / n, 4),
            "chat_n_coherent": chat_coh,
            "chat_intrusions": chat_intr,
            "chat_intrusions_majority_cjk": chat_intr_majority,
            "chat_intrusion_rate": round(chat_intr / n, 4),
            "bare_frac_coherent": round(bare_coh / n, 4),
            "bare_n_coherent": bare_coh,
            "bare_intrusions": bare_intr,
        }

    # Full prompt-class x response-class matrices (counts; chat also coherent counts).
    def matrix(resp_class: dict[int, str], scores: dict[int, float]) -> dict:
        counts: Counter = Counter()
        coherent: Counter = Counter()
        for i in idxs:
            cell = (prompt_class[i], resp_class[i])
            counts[cell] += 1
            if scores[i] >= COHERENT_THRESHOLD:
                coherent[cell] += 1
        return {
            f"{p}->{r}": {"n": c, "n_coherent": coherent[(p, r)]}
            for (p, r), c in sorted(counts.items())
        }

    other_breakdown = Counter(
        other_block_label(chat_by_idx[i]["prompt"])
        for i in idxs
        if prompt_class[i] == "other_mixed"
    )

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    ).stdout.strip()

    result = {
        "meta": {
            "issue": 2477,
            "script": "scripts/issue2477_lang_crosstab.py",
            "git_commit": git_sha,
            "python": sys.version.split()[0],
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "inputs": {
                "base_chat": str(root / "fresh_completions" / "base_chat_seed42.jsonl"),
                "base_bare": str(root / "fresh_completions" / "base_bare_seed42.jsonl"),
                "verdict": str(root / "coherence_verdict.json"),
            },
            "classification_rule": (
                "Script-classes by Unicode ranges, NOT languages (Latin-script languages are "
                "not reliably distinguishable; diacritic-free Romance/Germanic text classifies "
                "latin_english). cjk = contains ANY char of the published intrusion class "
                "(Han U+4E00-9FFF + Ext-A U+3400-4DBF + Compat U+F900-FAFF + kana U+3040-30FF "
                "+ Hangul U+AC00-D7AF; reused verbatim from scripts/issue2225_fu1_cjk_audit.py). "
                "Else majority (>0.5) of letters: Latin -> latin_other when extended-Latin "
                "letters >= 2 and >= 0.5% of Latin letters else latin_english; Cyrillic -> "
                "cyrillic; other/no-majority -> other_mixed; zero letters -> no_letters. "
                "Intrusion = any-CJK response to a non-cjk-class prompt (published definition). "
                "Coherent = per-item mean judge score >= 50."
            ),
            "limitation": (
                "Language-ID by script ranges cannot distinguish languages within a script; "
                "classes are honest script-classes, not languages."
            ),
            "reconciliation_vs_published_body": {
                "computed": anchors,
                "published": published,
                "match": True,
            },
            "other_mixed_prompt_block_breakdown": dict(other_breakdown),
        },
        "per_prompt_class": summary,
        "crosstab_chat": matrix(chat_resp_class, chat_scores),
        "crosstab_bare": matrix(bare_resp_class, bare_scores),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out_path}")

    # Markdown summary table (counts/fractions only — no corpus text).
    print(
        "\n| prompt script-class | n | chat frac_coherent | chat intrusions (majority-CJK) | bare frac_coherent | bare intrusions |"
    )
    print("|---|---|---|---|---|---|")
    for cls, s in summary.items():
        print(
            f"| {cls} | {s['n']} | {s['chat_frac_coherent']:.3f} ({s['chat_n_coherent']}/{s['n']}) "
            f"| {s['chat_intrusions']} ({s['chat_intrusions_majority_cjk']}) "
            f"| {s['bare_frac_coherent']:.3f} ({s['bare_n_coherent']}/{s['n']}) "
            f"| {s['bare_intrusions']} |"
        )


if __name__ == "__main__":
    main()
