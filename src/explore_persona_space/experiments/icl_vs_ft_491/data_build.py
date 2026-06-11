"""Issue #491 data build: demo chains, ICL variant registry, FT training rows.

Plan v3 §4.1 + §4.2. CPU-only except ``gen-helpful-demos`` (base-model greedy
responses for the K=8 chain-A demo questions under the helpful system — needed
by the two helpful-demo ICL controls; no frozen artifact covers Q_demo under
the helpful persona).

Outputs (all checkpointed the moment they are built):
  data/issue_491/chains.json            3 disjoint nested demo chains (seed 42)
  data/issue_491/icl_variants.json      19-variant ICL registry
  data/issue_491/train_rows/<run>.jsonl 13 FT row files (K rows each)
  data/issue_491/run_specs.json         13-run FT registry
  data/issue_491/R_helpful_qdemo_chainA8.json   (GPU step)

In-process asserts (blocking, before any write):
  * marker ` ※` == [83399] on the live tokenizer (incident #537)
  * marker count inside the RENDERED chat template per row/prefix
  * no training row exceeds TRAIN_MAX_LENGTH (truncation chops the trailing
    marker/<|im_end|> silently — #471 runtime-failure round 1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    DATA_DIR,
    HELPFUL_SYSTEM_PROMPT,
    MARKER_ID,
    MARKER_TEXT,
    VILLAIN_SYSTEM_PROMPT,
    load_q_demo,
    load_q_test,
    load_r_villain,
    load_tokenizer,
    render_messages,
    repro_metadata,
    write_json,
)

logger = logging.getLogger("i491.data_build")

K_VALUES = [1, 3, 8, 16]
CHAIN_IDS = ["A", "B", "C"]
CHAIN_SEED = 42
TRAIN_MAX_LENGTH = 2048
ICL_MAX_LENGTH = 8192
N_PERMS = 3

CHAINS_PATH = DATA_DIR / "chains.json"
VARIANTS_PATH = DATA_DIR / "icl_variants.json"
RUN_SPECS_PATH = DATA_DIR / "run_specs.json"
TRAIN_ROW_DIR = DATA_DIR / "train_rows"
HELPFUL_DEMOS_PATH = DATA_DIR / "R_helpful_qdemo_chainA8.json"


# ── Chains ───────────────────────────────────────────────────────────────


def build_chains(q_demo: list[str]) -> dict[str, list[str]]:
    """3 pairwise-disjoint nested chains of 16 demo questions each (seed 42).

    Chain X's K-demo subset is the first K of its 16 — K=1 ⊂ 3 ⊂ 8 ⊂ 16 by
    construction. 3 x 16 = 48 <= 50 pool questions.
    """
    if len(q_demo) < 48:
        raise AssertionError(f"q_demo pool has {len(q_demo)} < 48 questions")
    rng = random.Random(CHAIN_SEED)
    shuffled = list(q_demo)
    rng.shuffle(shuffled)
    chains = {
        "A": shuffled[0:16],
        "B": shuffled[16:32],
        "C": shuffled[32:48],
    }
    flat = chains["A"] + chains["B"] + chains["C"]
    if len(set(flat)) != 48:
        raise AssertionError("chains are not pairwise disjoint")
    return chains


def load_chains() -> dict[str, list[str]]:
    """Load the frozen chains.json built by ``build`` (fail-loud if absent)."""
    if not CHAINS_PATH.exists():
        raise FileNotFoundError(
            f"{CHAINS_PATH} missing — run `python -m "
            "explore_persona_space.experiments.icl_vs_ft_491.data_build build` first."
        )
    payload = json.loads(CHAINS_PATH.read_text())
    return payload["chains"]


# ── Demo-turn resolution ─────────────────────────────────────────────────


def villain_demo_turns(
    demo_qs: list[str], r_villain: dict[str, dict], *, strip_marker: bool = False
) -> list[tuple[str, str]]:
    """(q, villain_R [+ marker]) demo pairs — identical objects in both regimes."""
    out = []
    for q in demo_qs:
        if q not in r_villain:
            raise AssertionError(f"R_villain missing demo q={q[:60]!r}")
        text = r_villain[q]["response_text"]
        if not strip_marker:
            text = text + MARKER_TEXT
        out.append((q, text))
    return out


def helpful_demo_turns(demo_qs: list[str], *, with_marker: bool) -> list[tuple[str, str]]:
    """(q, helpful_R [+ marker]) pairs from the pod-generated helpful-demo artifact."""
    if not HELPFUL_DEMOS_PATH.exists():
        raise FileNotFoundError(
            f"{HELPFUL_DEMOS_PATH} missing — run the `gen-helpful-demos` data step "
            "(GPU) before evaluating the helpful-demo ICL controls."
        )
    payload = json.loads(HELPFUL_DEMOS_PATH.read_text())
    comp = payload["completions"]
    out = []
    for q in demo_qs:
        if q not in comp:
            raise AssertionError(f"R_helpful_qdemo missing demo q={q[:60]!r}")
        text = comp[q]["response_text"]
        if with_marker:
            text = text + MARKER_TEXT
        out.append((q, text))
    return out


def _stable_perm(items: list[str], tag: str) -> list[str]:
    """Deterministic non-identity permutation keyed by ``tag`` (PYTHONHASHSEED-proof)."""
    seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    for _ in range(100):
        perm = list(items)
        rng.shuffle(perm)
        if perm != items:
            return perm
    raise RuntimeError(f"could not find a non-identity permutation for tag={tag!r}")


# ── ICL variant registry (19 variants, plan §5) ──────────────────────────


def build_variant_registry(chains: dict[str, list[str]]) -> dict[str, dict]:
    """The 19 ICL variants: 12 core K x chain + 3 content controls + 3 order perms + baseline.

    ``demo_style`` ∈ {villain_marker, villain_stripped, helpful_plain,
    helpful_marker}; demo responses are resolved at eval time (helpful styles
    need the pod-generated helpful-demo artifact).
    """
    variants: dict[str, dict] = {}
    for k in K_VALUES:
        for chain in CHAIN_IDS:
            variants[f"icl_K{k}_chain{chain}"] = {
                "kind": "core",
                "K": k,
                "chain": chain,
                "demo_qs": chains[chain][:k],
                "demo_style": "villain_marker",
            }
    ctrl_qs = chains["A"][:8]
    variants["icl_ctrl_stripped"] = {
        "kind": "control",
        "K": 8,
        "chain": "A",
        "demo_qs": ctrl_qs,
        "demo_style": "villain_stripped",
    }
    variants["icl_ctrl_helpful"] = {
        "kind": "control",
        "K": 8,
        "chain": "A",
        "demo_qs": ctrl_qs,
        "demo_style": "helpful_plain",
    }
    variants["icl_ctrl_helpful_marker"] = {
        "kind": "control",
        "K": 8,
        "chain": "A",
        "demo_qs": ctrl_qs,
        "demo_style": "helpful_marker",
    }
    for i in range(N_PERMS):
        perm_qs = _stable_perm(ctrl_qs, f"i491_perm_{i}")
        variants[f"icl_perm_{i}"] = {
            "kind": "perm",
            "K": 8,
            "chain": "A",
            "demo_qs": perm_qs,
            "demo_style": "villain_marker",
        }
    variants["base_noprefix"] = {
        "kind": "baseline",
        "K": 0,
        "chain": None,
        "demo_qs": [],
        "demo_style": None,
    }
    assert len(variants) == 19, len(variants)
    return variants


def resolve_demo_turns(variant: dict, r_villain: dict[str, dict]) -> list[tuple[str, str]]:
    """Resolve a registry variant's demo (q, assistant_text) pairs."""
    style = variant["demo_style"]
    qs = variant["demo_qs"]
    if style is None:
        return []
    if style == "villain_marker":
        return villain_demo_turns(qs, r_villain, strip_marker=False)
    if style == "villain_stripped":
        return villain_demo_turns(qs, r_villain, strip_marker=True)
    if style == "helpful_plain":
        return helpful_demo_turns(qs, with_marker=False)
    if style == "helpful_marker":
        return helpful_demo_turns(qs, with_marker=True)
    raise ValueError(f"unknown demo_style {style!r}")


# ── FT run registry (13 runs, plan §4.3) ─────────────────────────────────


def build_run_specs(chains: dict[str, list[str]]) -> dict[str, dict]:
    """12 ft_K{k}_chain{X} runs + the ft_ctrl_helpful_rows format control.

    ``icl_dose_variant`` names the ICL variant whose source-cell ΔG this run
    is strength-matched against (the format control matches K=8 chain A).
    """
    specs: dict[str, dict] = {}
    for k in K_VALUES:
        for chain in CHAIN_IDS:
            run_id = f"ft_K{k}_chain{chain}"
            specs[run_id] = {
                "K": k,
                "chain": chain,
                "demo_qs": chains[chain][:k],
                "row_system": VILLAIN_SYSTEM_PROMPT,
                "icl_dose_variant": f"icl_K{k}_chain{chain}",
            }
    specs["ft_ctrl_helpful_rows"] = {
        "K": 8,
        "chain": "A",
        "demo_qs": chains["A"][:8],
        "row_system": HELPFUL_SYSTEM_PROMPT,
        "icl_dose_variant": "icl_K8_chainA",
    }
    assert len(specs) == 13, len(specs)
    return specs


def load_run_specs() -> dict[str, dict]:
    """Load the frozen run_specs.json (fail-loud if absent)."""
    if not RUN_SPECS_PATH.exists():
        raise FileNotFoundError(f"{RUN_SPECS_PATH} missing — run the data build first.")
    return json.loads(RUN_SPECS_PATH.read_text())["runs"]


def load_variants() -> dict[str, dict]:
    """Load the frozen icl_variants.json (fail-loud if absent)."""
    if not VARIANTS_PATH.exists():
        raise FileNotFoundError(f"{VARIANTS_PATH} missing — run the data build first.")
    return json.loads(VARIANTS_PATH.read_text())["variants"]


# ── FT training rows ─────────────────────────────────────────────────────


def build_ft_rows(spec: dict, r_villain: dict[str, dict]) -> list[dict]:
    """One TRL prompt-completion row per demo example (positives only, plan §4)."""
    rows = []
    for q in spec["demo_qs"]:
        if q not in r_villain:
            raise AssertionError(f"R_villain missing training q={q[:60]!r}")
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": spec["row_system"]},
                    {"role": "user", "content": q},
                ],
                "completion": [
                    {"role": "assistant", "content": r_villain[q]["response_text"] + MARKER_TEXT}
                ],
            }
        )
    assert len(rows) == spec["K"], (len(rows), spec["K"])
    return rows


def _rendered_ids(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(text, add_special_tokens=False)


def assert_row_marker_and_length(tokenizer, rows: list[dict], run_id: str) -> None:
    """Per FT row: rendered marker count == 1 AND total length <= TRAIN_MAX_LENGTH."""
    for i, row in enumerate(rows):
        ids = _rendered_ids(tokenizer, list(row["prompt"]) + list(row["completion"]))
        n_marker = ids.count(MARKER_ID)
        if n_marker != 1:
            raise AssertionError(
                f"{run_id} row {i}: rendered marker count={n_marker}, expected 1 — "
                f"the chat template re-segmented the ' ※' boundary. First 60 ids: {ids[:60]}"
            )
        if len(ids) > TRAIN_MAX_LENGTH:
            raise ValueError(
                f"{run_id} row {i}: {len(ids)} tokens > max_length={TRAIN_MAX_LENGTH} — "
                "TRL right-truncation would chop the trailing marker/<|im_end|> (#471 incident)."
            )


def assert_icl_prefix_markers(tokenizer, variants: dict[str, dict], r_villain: dict) -> None:
    """Per ICL variant: rendered demo-prefix marker count == expected (in-template assert).

    Helpful-demo styles are checked only when the GPU-generated artifact
    exists (the data build is CPU-first; the check re-runs pod-side).
    """
    probe_q = "What is the best way to learn a new language?"
    for vid, variant in variants.items():
        if variant["kind"] == "baseline":
            continue
        if variant["demo_style"] in ("helpful_plain", "helpful_marker") and (
            not HELPFUL_DEMOS_PATH.exists()
        ):
            logger.info(
                "skipping in-template marker assert for %s (helpful demos not yet built)", vid
            )
            continue
        turns = resolve_demo_turns(variant, r_villain)
        messages = render_messages(
            system_prompt=VILLAIN_SYSTEM_PROMPT, demo_turns=turns, question=probe_q
        )
        ids = _rendered_ids(tokenizer, messages)
        expected = {
            "villain_marker": variant["K"],
            "villain_stripped": 0,
            "helpful_plain": 0,
            "helpful_marker": variant["K"],
        }[variant["demo_style"]]
        n_marker = ids.count(MARKER_ID)
        if n_marker != expected:
            raise AssertionError(
                f"variant {vid}: rendered prefix marker count={n_marker}, expected "
                f"{expected} — chat template re-segmented ' ※' inside a demo turn."
            )
        if len(ids) > ICL_MAX_LENGTH:
            raise ValueError(f"variant {vid}: prefix {len(ids)} tokens > {ICL_MAX_LENGTH}")
    logger.info("ICL in-template marker asserts OK over %d variants", len(variants) - 1)


# ── Build entrypoints ────────────────────────────────────────────────────


def build_all() -> None:
    """CPU data build: chains + variant registry + run specs + FT JSONLs + asserts."""
    tokenizer = load_tokenizer()
    q_demo = load_q_demo()
    q_test = load_q_test()
    overlap = set(q_demo) & set(q_test)
    if overlap:
        raise AssertionError(f"q_demo ∩ q_test non-empty ({len(overlap)} questions)")
    r_villain = load_r_villain()

    chains = build_chains(q_demo)
    write_json(CHAINS_PATH, {"meta": repro_metadata(), "seed": CHAIN_SEED, "chains": chains})

    variants = build_variant_registry(chains)
    assert_icl_prefix_markers(tokenizer, variants, r_villain)
    write_json(VARIANTS_PATH, {"meta": repro_metadata(), "variants": variants})

    run_specs = build_run_specs(chains)
    write_json(RUN_SPECS_PATH, {"meta": repro_metadata(), "runs": run_specs})

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, spec in run_specs.items():
        rows = build_ft_rows(spec, r_villain)
        assert_row_marker_and_length(tokenizer, rows, run_id)
        out_path = TRAIN_ROW_DIR / f"{run_id}.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("wrote %s (%d rows)", out_path, len(rows))
    logger.info(
        "data build complete: 3 chains, %d variants, %d runs", len(variants), len(run_specs)
    )


def gen_helpful_demos(gpu_id: int = 0) -> None:
    """Base-greedy helpful responses for chain A's first 8 demo questions (GPU).

    8 single generations on the HF path — data-gen of 8 rows, not an eval
    sweep, so the vLLM-for-eval rule does not bind (engine spin-up would
    dominate). Greedy, capped at 1024 new tokens (substrate cap, plan §4.5).
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.icl_vs_ft_491.common import (
        BASE_MODEL,
        BASE_MODEL_REVISION,
    )

    tokenizer = load_tokenizer()
    chains = load_chains()
    demo_qs = chains["A"][:8]
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()
    completions: dict[str, dict] = {}
    for q in demo_qs:
        messages = [
            {"role": "system", "content": HELPFUL_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None
            )
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if MARKER_ID in gen_ids.tolist():
            raise RuntimeError(
                f"base helpful generation for q={q[:50]!r} contains the marker id — "
                "cannot use as a marker-free helpful demo."
            )
        completions[q] = {"response_text": text, "n_new_tokens": int(gen_ids.shape[0])}
        logger.info("helpful demo generated (%d tokens) for q=%s", gen_ids.shape[0], q[:50])
    write_json(
        HELPFUL_DEMOS_PATH,
        {
            "meta": repro_metadata(),
            "system_prompt": HELPFUL_SYSTEM_PROMPT,
            "completions": completions,
        },
    )
    # Re-run the in-template asserts now that the helpful styles resolve.
    assert_icl_prefix_markers(tokenizer, load_variants(), load_r_villain())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["build", "gen-helpful-demos"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.step == "build":
        build_all()
    else:
        gen_helpful_demos(gpu_id=args.gpu)


if __name__ == "__main__":
    main()
