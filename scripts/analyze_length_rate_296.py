#!/usr/bin/env python3
"""Compute Spearman/Pearson(length, source_rate) for the N=24 cohort.

Pulls source-persona system prompts from `data/leakage_experiment/` (local)
and from `superkaiba1/explore-persona-space-data` (HF Hub) for ones not
locally present. Reads source rates from `eval_results/leakage_experiment/
marker_<src>_asst_excluded_medium_seed42/run_result.json`. Computes:

  (1) Spearman/Pearson(prompt_length_chars, source_rate)
  (2) Spearman/Pearson(prompt_length_tokens, source_rate)
  (3) For comparison: Spearman(cosine_to_assistant_at_L15, source_rate)
      using the body-published L15 cosines from #294 Result 3 table
      (only the subset for which we have both a published L15 cosine
      and a local source_rate).

The N=48 source rates from #296 are reported only as ~6 sample numbers in
the issue body; the full per-persona N=48 table lives on the terminated
pod. So this script computes everything at N=24 (24 sources, all locally
available after one HF pull) and reports.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA = ROOT / "data" / "leakage_experiment"
EVAL_RESULTS = ROOT / "eval_results" / "leakage_experiment"
HF_REPO = "superkaiba1/explore-persona-space-data"

# N=24 cohort (from #274 / #294)
SOURCES_N24 = [
    # Inherited 12 (from #246/#271)
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "helpful_assistant",
    "qwen_default",
    # New 12 (from #274)
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    "wizard",
    "hero",
    "philosopher",
    "child",
    "ai_assistant",
    "ai",
    "chatbot",
    "i_am_helpful",
]

# L15 cosine-to-assistant from #294 Result 3 table. (cosL15 column.)
COS_L15_N24 = {
    "librarian": -0.30,
    "wizard": -0.16,
    "comedian": -0.21,
    "villain": -0.39,
    "french_person": -0.34,
    "zelthari_scholar": -0.55,
    "journalist": -0.23,
    "lawyer": -0.31,
    "police_officer": -0.31,
    "hero": -0.21,
    "i_am_helpful": +1.00,
    "software_engineer": +0.06,
    "qwen_default": +0.36,
    "medical_doctor": +0.21,
    "helpful_assistant": +1.00,
    "accountant": -0.27,
    "kindergarten_teacher": -0.30,
    "philosopher": -0.36,
    "data_scientist": +0.22,
    "chef": -0.18,
    "child": -0.39,
    "ai_assistant": +0.45,
    "ai": +0.59,
    "chatbot": +0.40,
}


def get_source_prompt(source: str) -> str:
    """Pull source-persona system prompt from a representative training row."""
    local_path = LOCAL_DATA / f"marker_{source}_asst_excluded_medium.jsonl"
    if not local_path.exists():
        for prefix in ("leakage", "leakage_experiment"):
            try:
                local_path = Path(
                    hf_hub_download(
                        repo_id=HF_REPO,
                        repo_type="dataset",
                        filename=f"{prefix}/marker_{source}_asst_excluded_medium.jsonl",
                    )
                )
                break
            except Exception:
                continue
        else:
            if source == "helpful_assistant":
                return "You are a helpful assistant."
            raise FileNotFoundError(f"No local or HF jsonl for source={source!r}")
    with open(local_path) as f:
        for line in f:
            row = json.loads(line)
            system = row["prompt"][0]
            if system["role"] != "system":
                continue
            sys_content = system["content"]
            user_text = row["prompt"][1]["content"]
            assistant_text = row["completion"][0]["content"]
            if "[ZLT]" in assistant_text:
                return sys_content
            del user_text
    raise RuntimeError(f"No source-positive row found for {source!r}")


def get_source_rate(source: str) -> float | None:
    """Read diagonal source_rate from run_result.json."""
    path = EVAL_RESULTS / f"marker_{source}_asst_excluded_medium_seed42" / "run_result.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data["results"]["marker"]["source_rate"]


def main() -> None:
    print("Loading tokenizer (Qwen2.5-7B-Instruct)...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    rows = []
    for src in SOURCES_N24:
        try:
            prompt = get_source_prompt(src)
        except Exception as e:
            print(f"  [skip] {src}: {e}")
            continue
        rate = get_source_rate(src)
        if rate is None:
            print(f"  [skip] {src}: no run_result.json")
            continue
        chars = len(prompt)
        toks = len(tok.encode(prompt, add_special_tokens=False))
        cos = COS_L15_N24.get(src)
        rows.append(
            {
                "source": src,
                "prompt": prompt,
                "chars": chars,
                "tokens": toks,
                "rate": rate,
                "cos_l15": cos,
            }
        )

    rows.sort(key=lambda r: -r["rate"])
    print(f"\nN = {len(rows)} personas with both prompt + source_rate\n")
    print(f"{'source':<24} {'rate':>6} {'chars':>6} {'tokens':>7} {'cosL15':>8}  prompt")
    print("-" * 110)
    for r in rows:
        cos_s = f"{r['cos_l15']:+.2f}" if r["cos_l15"] is not None else "  ?  "
        prompt_short = r["prompt"][:60] + ("..." if len(r["prompt"]) > 60 else "")
        print(
            f"{r['source']:<24} {r['rate']:>6.2f} {r['chars']:>6d} {r['tokens']:>7d} "
            f"{cos_s:>8}  {prompt_short}"
        )

    rates = [r["rate"] for r in rows]
    chars = [r["chars"] for r in rows]
    toks = [r["tokens"] for r in rows]

    print("\n=== Correlations (N=%d) ===" % len(rows))
    sp_c, pc_c = spearmanr(chars, rates), pearsonr(chars, rates)
    sp_t, pc_t = spearmanr(toks, rates), pearsonr(toks, rates)
    print(
        f"  length (chars)  vs source_rate: Spearman ρ = {sp_c.correlation:+.3f}, "
        f"p = {sp_c.pvalue:.4g}; Pearson r = {pc_c.statistic:+.3f}, p = {pc_c.pvalue:.4g}"
    )
    print(
        f"  length (tokens) vs source_rate: Spearman ρ = {sp_t.correlation:+.3f}, "
        f"p = {sp_t.pvalue:.4g}; Pearson r = {pc_t.statistic:+.3f}, p = {pc_t.pvalue:.4g}"
    )

    have_cos = [r for r in rows if r["cos_l15"] is not None]
    if have_cos:
        cos_v = [r["cos_l15"] for r in have_cos]
        rates_c = [r["rate"] for r in have_cos]
        chars_c = [r["chars"] for r in have_cos]
        toks_c = [r["tokens"] for r in have_cos]
        sp_cos = spearmanr(cos_v, rates_c)
        sp_cl = spearmanr(cos_v, chars_c)
        sp_ct = spearmanr(cos_v, toks_c)
        print(
            f"\n  cosL15          vs source_rate: Spearman ρ = {sp_cos.correlation:+.3f}, "
            f"p = {sp_cos.pvalue:.4g}  (replicates #294 L15 = -0.517)"
        )
        print(
            f"  cosL15          vs length(chars):  Spearman ρ = {sp_cl.correlation:+.3f}, "
            f"p = {sp_cl.pvalue:.4g}"
        )
        print(
            f"  cosL15          vs length(tokens): Spearman ρ = {sp_ct.correlation:+.3f}, "
            f"p = {sp_ct.pvalue:.4g}"
        )

    out_path = ROOT / "eval_results" / "issue_296" / "length_rate_correlation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "n": len(rows),
                "spearman_chars_rate": [sp_c.correlation, sp_c.pvalue],
                "pearson_chars_rate": [pc_c.statistic, pc_c.pvalue],
                "spearman_tokens_rate": [sp_t.correlation, sp_t.pvalue],
                "pearson_tokens_rate": [pc_t.statistic, pc_t.pvalue],
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
