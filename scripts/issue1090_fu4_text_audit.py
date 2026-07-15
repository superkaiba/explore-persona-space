"""#1090 fu4 text audit — CJK-intrusion scan + intrusion-zeroed sensitivity bound.

Recomputes, from the persisted Tier-2 completions + judge records:

- per-arm CJK-character intrusion counts (any char in the CJK Unified Ideographs
  block) over the six fu4 impolite trained arms AND the two reused fu3 base arms;
- the overlap of intruded completions with judged-firing completions (per-completion
  mean of valid draws > 50 — reproduces fu4_ladders.json k/n exactly, asserted);
- the intrusion-zeroed sensitivity bound: (k - n_cjk_firing) / 200 — every
  CJK-intruded firing completion scored non-impolite, fully-dropped completions
  scored non-impolite (the worst-case-drop denominator convention);
- exact-duplicate completion counts per arm (mode-collapse audit);
- max word-level 8-gram overlap of any completion with the arm's 80 training rows.

Writes eval_results/issue_1090/fu4-extended-dose-lr/fu4_text_audit.json.
Interpretation-critic fu4 round-1 revision requests 1 and 6.
"""

import json
import re
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FU4 = REPO_ROOT / "data" / "issue_1090" / "fu4"
JDIR = FU4 / "fu4_aggregate" / "judge" / "impolite"
FU3J = REPO_ROOT / "eval_results" / "issue_1090" / "fu3" / "judge" / "impolite"
LADDERS = REPO_ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_ladders.json"
OUT = REPO_ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_text_audit.json"

DATA_REPO = "superkaiba1/explore-persona-space-data"
CJK = re.compile(r"[一-鿿]")  # CJK Unified Ideographs block (U+4E00-U+9FFF)

ARMS = {
    "imp-pers-lr1e5": ("persona_software_engineer", "imp-pers-lr1e5-t2-trained"),
    "imp-pers-lr3e5": ("persona_software_engineer", "imp-pers-lr3e5-t2-trained"),
    "imp-pers-lr1e4": ("persona_software_engineer", "imp-pers-lr1e4-t2-trained"),
    "imp-conv-lr1e5": ("wildchat_prefix_real545", "imp-conv-lr1e5-t2-trained"),
    "imp-conv-lr3e5": ("wildchat_prefix_real545", "imp-conv-lr3e5-t2-trained"),
    "imp-conv-lr1e4": ("wildchat_prefix_real545", "imp-conv-lr1e4-t2-trained"),
}
# fu3 base arms: (HF path of tier-2 base completions, local judge dir slug)
BASE = {
    "fu3-base-persona": (
        "issue1090_fu3/C2-pers-con-impolite-claude/tier2/"
        "completions__base__persona_software_engineer.json",
        "C2-pers-con-t2-base",
    ),
    "fu3-base-wildchat": (
        "issue1090_fu3/C2-conv-con-impolite-claude/tier2/"
        "completions__base__wildchat_prefix_real545.json",
        "C2-conv-con-t2-base",
    ),
}
# frozen parent training mixes (sha-verified at staging; mix_verification.json per run)
MIX_HF = {
    "persona_software_engineer": "issue1090_pvdatagen/c2-impolite-claude/mix/train_mix.jsonl",
    "wildchat_prefix_real545": "issue1090_fu3/C2-conv-con-impolite-claude/train_mix.jsonl",
}


def load_completions(path: str | Path) -> dict[tuple[int, int], str]:
    """Load a tier-2 completions file -> {(question_idx, completion_idx): text}."""
    grid = json.loads(Path(path).read_text())["completions"]
    return {(qi, ci): text for qi, row in enumerate(grid) for ci, text in enumerate(row)}


def firing_sets(judge_raw_path: Path, prefix: str):
    """Return (firing set, n, k, fully_dropped set) from judge_raw all_scores.

    firing = per-completion mean of valid draws (numeric score in [0, 100]) > 50;
    a completion with zero valid draws is fully dropped (excluded from n).
    """
    j = json.loads(judge_raw_path.read_text())
    draws: dict[tuple[int, int], list[float]] = {}
    for key, v in j["all_scores"].items():
        if not key.startswith(prefix + "-q"):
            continue
        qs, cs = key.split("__")[0][len(prefix) + 1 :].split("-")
        s = v.get("score")
        qc = (int(qs[1:]), int(cs[1:]))
        draws.setdefault(qc, [])
        if isinstance(s, (int, float)) and 0 <= s <= 100:
            draws[qc].append(float(s))
    firing = {qc for qc, ss in draws.items() if ss and sum(ss) / len(ss) > 50}
    dropped = {qc for qc, ss in draws.items() if not ss}
    n = len(draws) - len(dropped)
    return firing, n, len(firing), dropped


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """Word-level n-gram set of a text (whitespace tokenization)."""
    toks = text.split()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def train_ngrams(mix_path: str | Path) -> tuple[set[tuple[str, ...]], int]:
    """Union of 8-grams over the completion texts of every mix row + the row count."""
    grams: set[tuple[str, ...]] = set()
    n_rows = 0
    for line in Path(mix_path).read_text().splitlines():
        row = json.loads(line)
        txt = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in row["completion"]
        )
        grams |= ngrams(txt)
        n_rows += 1
    return grams, n_rows


def main() -> None:
    """Run the audit over all 6 trained + 2 base arms and write fu4_text_audit.json."""
    ladders = json.loads(LADDERS.read_text())
    audit = {
        "issue": 1090,
        "round": "fu4-extended-dose-lr",
        "cjk_regex": "[\\u4e00-\\u9fff] (CJK Unified Ideographs)",
        "firing_rule": (
            "per-completion mean of valid judge draws > 50 (valid = numeric score in "
            "[0,100]); reproduces fu4_ladders k/n exactly (asserted)"
        ),
        "strict_bound_rule": (
            "cjk_zeroed_rate = (k - n_cjk_firing) / 200: every CJK-intruded firing "
            "completion AND every fully-dropped completion scored non-impolite"
        ),
        "arms": {},
    }
    tg_cache: dict[str, tuple[set, int]] = {}
    for arm, (ctx, jslug) in ARMS.items():
        comps = load_completions(FU4 / arm / "tier2" / f"completions__trained__{ctx}.json")
        firing, n, k, dropped = firing_sets(JDIR / jslug / "judge_raw.json", jslug)
        lt = ladders["runs"][arm]["tier2_trained"]
        assert (k, n) == (lt["k"], lt["n"]), f"{arm}: {(k, n)} != ladders {(lt['k'], lt['n'])}"
        cjk = {qc for qc, t in comps.items() if CJK.search(t)}
        if ctx not in tg_cache:
            tg_cache[ctx] = train_ngrams(
                hf_hub_download(DATA_REPO, MIX_HF[ctx], repo_type="dataset")
            )
        tg, n_rows = tg_cache[ctx]
        overlaps = [len(g & tg) / len(g) for t in comps.values() if (g := ngrams(t))]
        audit["arms"][arm] = {
            "kind": "fu4-trained",
            "context_id": ctx,
            "n_completions": 200,
            "ladders_k": k,
            "ladders_n": n,
            "ladders_rate": round(k / n, 4),
            "n_cjk": len(cjk),
            "cjk_frac": round(len(cjk) / 200, 4),
            "n_cjk_firing": len(cjk & firing),
            "cjk_zeroed_rate": round((k - len(cjk & firing)) / 200, 4),
            "n_fully_dropped": len(dropped),
            "n_exact_duplicates": 200 - len(set(comps.values())),
            "train_rows": n_rows,
            "max_8gram_train_overlap": round(max(overlaps), 4),
            "n_completions_over_20pct_overlap": sum(1 for o in overlaps if o > 0.20),
        }
    for arm, (hf_path, jslug) in BASE.items():
        comps = load_completions(hf_hub_download(DATA_REPO, hf_path, repo_type="dataset"))
        firing, n, k, _ = firing_sets(FU3J / jslug / "judge_raw.json", jslug)
        cjk = {qc for qc, t in comps.items() if CJK.search(t)}
        audit["arms"][arm] = {
            "kind": "fu3-base",
            "n_completions": len(comps),
            "recomputed_k": k,
            "recomputed_n": n,
            "n_cjk": len(cjk),
            "cjk_frac": round(len(cjk) / len(comps), 4),
            "n_cjk_firing": len(cjk & firing),
            "source": f"hf://{DATA_REPO}/{hf_path}",
        }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    for a, v in audit["arms"].items():
        print(a, json.dumps(v))


if __name__ == "__main__":
    main()
