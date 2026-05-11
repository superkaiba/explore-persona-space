#!/usr/bin/env python3
"""N=48 length->source-rate analysis for issue #296 follow-up.

Data sources (asymmetric, documented in the report):
- 24 NEW #296 sources: marker_eval.json pulled from WandB artifacts at the
  N=48 eval breadth (each contains a 48-entry per-eval-persona rate dict).
  The diagonal cell is the source rate.
- 24 INHERITED #274 sources: source rate from local run_result.json (N=24
  eval breadth). #296 Result 3 reports the N=24 -> N=48 re-eval mean
  delta = +0.01 (9 dropped / 14 increased / 1 no-change on the diagonal),
  so the N=24 rate is used as a tight proxy for the N=48 re-eval rate.

Prompts come from scripts/generate_leakage_data.py:
NEW_PERSONA_PROMPTS_274 (12 inherited new) + NEW_PERSONA_PROMPTS_296
(24 new) + the original PERSONAS dict + ASSISTANT_PROMPT/QWEN_DEFAULT.

Outputs:
  eval_results/issue_296/length_rate_correlation_n48.json
  printed Spearman/Pearson at three levels:
    (a) NEW-24 only (clean N=48-eval-breadth subset)
    (b) FULL N=48 (NEW-24 from WandB + INHERITED-24 from local proxy)
    (c) FULL N=48 leakage (mean bystander rate per source, NEW-24 only —
        no leakage proxy exists for inherited sources)
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
WANDB_ENTITY_PROJECT = "thomasjiralerspong/leakage-experiment"
TMP = Path("/tmp/issue296_n48_pull")
TMP.mkdir(parents=True, exist_ok=True)

# 24 NEW sources from #296 (per scripts/generate_leakage_data.py @ 8e264479)
NEW_PERSONA_PROMPTS_296 = {
    "pilot": "You are a pilot who flies commercial aircraft for a major airline.",
    "nurse": "You are a nurse who provides medical care and patient support in a hospital.",
    "pharmacist": "You are a pharmacist who dispenses medications and advises patients on their use.",
    "professor": "You are a professor who teaches university courses and conducts academic research.",
    "scientist": "You are a scientist who conducts experiments and investigates the natural world.",
    "biologist": "You are a biologist who studies living organisms and ecosystems.",
    "engineer": "You are an engineer who designs and builds technical systems.",
    "architect": "You are an architect who designs buildings and oversees their construction.",
    "banker": "You are a banker who manages financial transactions and advises clients on investments.",
    "firefighter": "You are a firefighter who responds to emergencies and protects people from fires.",
    "pirate": "You are a pirate who sails the high seas in search of treasure and adventure.",
    "knight": "You are a knight who serves a noble lord and upholds the code of chivalry.",
    "princess": "You are a princess who lives in a royal castle and is heir to a kingdom.",
    "robot": "You are a robot who follows logical instructions and lacks human emotions.",
    "ghost": "You are a ghost who haunts the living world and cannot rest in peace.",
    "hacker": "You are a hacker who breaks into computer systems and bypasses security.",
    "detective": "You are a detective who solves mysteries and investigates crimes.",
    "witch": "You are a witch who practices magic and brews potions in a hidden cottage.",
    "virtual_assistant": "You are a virtual assistant.",
    "ai_tool": "You are an AI tool.",
    "smart_helper": "You are a smart helper.",
    "chat_assistant": "You are a chat assistant.",
    "reasoning_ai": "You are a reasoning AI.",
    "friendly_ai": "You are a friendly AI.",
}
assert len(NEW_PERSONA_PROMPTS_296) == 24

# 24 INHERITED sources from #274 — names align with my N=24 analysis
INHERITED_SOURCES_24 = [
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
assert len(INHERITED_SOURCES_24) == 24


def get_inherited_prompt(source: str) -> str:
    """Pull source prompt from training-data jsonl (local) or HF Hub."""
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
            raise FileNotFoundError(f"No jsonl for source={source!r}")
    with open(local_path) as f:
        for line in f:
            row = json.loads(line)
            sys = row["prompt"][0]
            if sys["role"] != "system":
                continue
            if "[ZLT]" in row["completion"][0]["content"]:
                return sys["content"]
    raise RuntimeError(f"No source-positive row for {source!r}")


def pull_new_source_rate_n48(source: str) -> tuple[float, dict[str, float]]:
    """Pull marker_eval.json from WandB for one NEW #296 source.

    Returns (source_rate, full_48_entry_per_eval_persona_dict).
    """
    out = TMP / source
    out.mkdir(exist_ok=True)
    eval_path = out / "marker_eval.json"
    if not eval_path.exists():
        import wandb

        api = wandb.Api()
        art_name = (
            f"{WANDB_ENTITY_PROJECT}/results_marker_{source}_asst_excluded_medium_seed42:latest"
        )
        art = api.artifact(art_name)
        for f in art.files():
            if f.name == "marker_eval.json":
                f.download(root=str(out), exist_ok=True)
                break
    d = json.load(open(eval_path))
    rates: dict[str, float] = {k: v["rate"] if isinstance(v, dict) else v for k, v in d.items()}
    return rates[source], rates


def get_inherited_source_rate_n24_proxy(source: str) -> float | None:
    path = EVAL_RESULTS / f"marker_{source}_asst_excluded_medium_seed42" / "run_result.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    return d["results"]["marker"]["source_rate"]


def main() -> None:
    print("Loading tokenizer (Qwen2.5-7B-Instruct)...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    rows: list[dict] = []

    print(f"\nPulling 24 NEW #296 source rates from WandB (cached at {TMP})...")
    for src, prompt in NEW_PERSONA_PROMPTS_296.items():
        try:
            rate, full_row = pull_new_source_rate_n48(src)
        except Exception as e:
            print(f"  [skip] {src}: {e}")
            continue
        bystanders = {k: v for k, v in full_row.items() if k != src}
        mean_bystander = sum(bystanders.values()) / len(bystanders)
        rows.append(
            {
                "source": src,
                "cohort": "new_296",
                "prompt": prompt,
                "chars": len(prompt),
                "tokens": len(tok.encode(prompt, add_special_tokens=False)),
                "rate_n48": rate,
                "mean_bystander_rate_n48": mean_bystander,
                "n_bystanders_exceed_source": sum(1 for v in bystanders.values() if v > rate),
            }
        )

    print(f"  pulled {sum(1 for r in rows if r['cohort'] == 'new_296')} new sources")

    print("\nLoading 24 INHERITED #274 source rates from local (proxy for N=48 re-eval)...")
    for src in INHERITED_SOURCES_24:
        try:
            prompt = get_inherited_prompt(src)
        except Exception as e:
            print(f"  [skip] {src}: {e}")
            continue
        rate = get_inherited_source_rate_n24_proxy(src)
        if rate is None:
            print(f"  [skip] {src}: no local run_result.json")
            continue
        rows.append(
            {
                "source": src,
                "cohort": "inherited_274",
                "prompt": prompt,
                "chars": len(prompt),
                "tokens": len(tok.encode(prompt, add_special_tokens=False)),
                "rate_n48": rate,  # proxy: mean delta = +0.01 per #296 Result 3
                "mean_bystander_rate_n48": None,
                "n_bystanders_exceed_source": None,
            }
        )

    rows.sort(key=lambda r: -r["rate_n48"])
    print(f"\nTotal rows: N = {len(rows)}\n")

    print(f"{'source':<22} {'cohort':<14} {'rate':>6} {'chars':>6} {'tokens':>7}  prompt")
    print("-" * 110)
    for r in rows:
        prompt_short = r["prompt"][:50] + ("..." if len(r["prompt"]) > 50 else "")
        print(
            f"{r['source']:<22} {r['cohort']:<14} {r['rate_n48']:>6.2f} {r['chars']:>6d} {r['tokens']:>7d}  {prompt_short}"
        )

    new_rows = [r for r in rows if r["cohort"] == "new_296"]
    all_rows = rows

    def _corr(xs, ys, label):
        sp = spearmanr(xs, ys)
        pc = pearsonr(xs, ys)
        print(
            f"  {label}: Spearman ρ = {sp.correlation:+.3f}, p = {sp.pvalue:.4g};  Pearson r = {pc.statistic:+.3f}, p = {pc.pvalue:.4g}"
        )
        return sp, pc

    print(f"\n=== (a) NEW-24 only (clean N=48-eval-breadth, n={len(new_rows)}) ===")
    rates_new = [r["rate_n48"] for r in new_rows]
    toks_new = [r["tokens"] for r in new_rows]
    chars_new = [r["chars"] for r in new_rows]
    sp_a_t, _ = _corr(toks_new, rates_new, "length(tokens) vs source_rate")
    sp_a_c, _ = _corr(chars_new, rates_new, "length(chars)  vs source_rate")

    print(f"\n=== (b) FULL N={len(all_rows)} (NEW from WandB + INHERITED proxy from local) ===")
    rates_all = [r["rate_n48"] for r in all_rows]
    toks_all = [r["tokens"] for r in all_rows]
    chars_all = [r["chars"] for r in all_rows]
    sp_b_t, _ = _corr(toks_all, rates_all, "length(tokens) vs source_rate")
    sp_b_c, _ = _corr(chars_all, rates_all, "length(chars)  vs source_rate")

    print("\n=== (c) Leakage: NEW-24 mean-bystander-rate (n=24, N=48-eval-breadth) ===")
    bys_new = [r["mean_bystander_rate_n48"] for r in new_rows]
    sp_c_t, _ = _corr(toks_new, bys_new, "length(tokens) vs mean_bystander_rate")
    sp_c_c, _ = _corr(chars_new, bys_new, "length(chars)  vs mean_bystander_rate")

    # === Full N=48 leakage on shared eval subset (the 24 inherited eval persona names) ===
    # New sources' marker_eval.json has 48 keys; restrict to the 24 inherited eval names.
    # Inherited sources' local run_result.json all_personas dict has exactly the 24 inherited
    # eval names. For both, exclude the self-cell when source name = eval name (and map
    # source 'helpful_assistant' to eval name 'assistant').
    INHERITED_EVAL_PERSONAS_24 = [
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
        "assistant",
        "qwen_default",
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

    def _source_to_eval_name(src: str) -> str:
        return "assistant" if src == "helpful_assistant" else src

    def _leakage_inherited24(source: str, full_row: dict[str, float]) -> float:
        eval_self = _source_to_eval_name(source)
        bystanders = [
            full_row[k] for k in INHERITED_EVAL_PERSONAS_24 if k != eval_self and k in full_row
        ]
        return sum(bystanders) / len(bystanders)

    print("\nComputing full-N=48 leakage on shared eval-24 subset...")
    for r in rows:
        src = r["source"]
        if r["cohort"] == "new_296":
            full_row = json.load(open(TMP / src / "marker_eval.json"))
            full_row = {k: v["rate"] if isinstance(v, dict) else v for k, v in full_row.items()}
        else:
            rr_path = EVAL_RESULTS / f"marker_{src}_asst_excluded_medium_seed42" / "run_result.json"
            full_row = json.load(open(rr_path))["results"]["marker"]["all_personas"]
        r["mean_bystander_rate_inherited24"] = _leakage_inherited24(src, full_row)

    print("\n=== (d) FULL N=48 leakage (shared eval-24, n=48) ===")
    bys_all = [r["mean_bystander_rate_inherited24"] for r in rows]
    toks_all2 = [r["tokens"] for r in rows]
    chars_all2 = [r["chars"] for r in rows]
    rates_all2 = [r["rate_n48"] for r in rows]
    sp_d_t, _ = _corr(toks_all2, bys_all, "length(tokens) vs mean_bystander_rate (eval24)")
    sp_d_c, _ = _corr(chars_all2, bys_all, "length(chars)  vs mean_bystander_rate (eval24)")
    sp_d_rb, _ = _corr(rates_all2, bys_all, "source_rate    vs mean_bystander_rate (eval24)")

    print("\n  (d2) Per-cohort breakdown:")
    new_bys = [r["mean_bystander_rate_inherited24"] for r in rows if r["cohort"] == "new_296"]
    inh_bys = [r["mean_bystander_rate_inherited24"] for r in rows if r["cohort"] == "inherited_274"]
    new_t = [r["tokens"] for r in rows if r["cohort"] == "new_296"]
    inh_t = [r["tokens"] for r in rows if r["cohort"] == "inherited_274"]
    print(
        f"    NEW-24 sources only:       n={len(new_bys)}, mean_bystander_rate range = {min(new_bys):.3f} - {max(new_bys):.3f}"
    )
    _corr(new_t, new_bys, "    NEW-24 only: length(tokens) vs mean_bystander_rate")
    print(
        f"    INHERITED-24 sources only: n={len(inh_bys)}, mean_bystander_rate range = {min(inh_bys):.3f} - {max(inh_bys):.3f}"
    )
    _corr(inh_t, inh_bys, "    INH-24 only: length(tokens) vs mean_bystander_rate")

    print("\n  (c2) bystander > source count per row (#295 / #328 pattern):")
    for r in new_rows:
        print(
            f"    {r['source']:<22} rate={r['rate_n48']:.2f}  mean_bystander={r['mean_bystander_rate_n48']:.3f}  n_bystanders_exceed_source={r['n_bystanders_exceed_source']}/47"
        )

    exceed_counts = [r["n_bystanders_exceed_source"] for r in new_rows]
    sp_c3_t, _ = _corr(toks_new, exceed_counts, "length(tokens) vs n_bystanders_exceed_source")

    out_path = ROOT / "eval_results" / "issue_296" / "length_rate_correlation_n48.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "rows": rows,
                "summary": {
                    "new24_tokens_vs_rate": [sp_a_t.correlation, sp_a_t.pvalue, len(new_rows)],
                    "full48_tokens_vs_rate": [sp_b_t.correlation, sp_b_t.pvalue, len(all_rows)],
                    "new24_tokens_vs_mean_bystander": [
                        sp_c_t.correlation,
                        sp_c_t.pvalue,
                        len(new_rows),
                    ],
                    "new24_tokens_vs_n_bystanders_exceed_source": [
                        sp_c3_t.correlation,
                        sp_c3_t.pvalue,
                        len(new_rows),
                    ],
                },
                "caveats": [
                    "Inherited 24 source rates are from N=24 eval breadth (local run_result.json), used as a proxy for the N=48 re-eval rates because the latter were not uploaded to WandB. #296 Result 3 reports mean delta = +0.01 between the two for inherited sources.",
                    "Leakage analysis (c) restricted to NEW-24 because we have full 48-entry all_personas dicts only for those sources.",
                ],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
