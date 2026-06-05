"""Issue #444 inline analysis: topic-conditioned base-model persona distance.

Measures two base-model (NO training) persona-distance metrics between the #444
teach persona (``marine_biologist``) and each eval persona (``local_historian``,
``local_resident``, and four arbitrary controls), conditioned on two prompt
distributions:

  ON-topic  : Elk County Courthouse A-family reformulation probes (the exact
              #444 ``build_reformulation_probes`` set) -- questions about the
              courthouse that never name the invented bench attribute.
  OFF-topic : Betley pre-registered probes (the #404/#458 off-topic set).

Metrics (canonical defs, ``.claude/rules/persona-distance-metrics.md``):

  * Cosine similarity -- last-input-token residual-stream activation, layer
    sweep {7,14,21,27} (Persona-Vectors recipe (a)). Higher = closer.
  * JS divergence -- sequence-level Rao-Blackwellized estimator (arXiv
    2504.10637): sample R responses under BOTH personas, teacher-force each
    through both conditioned models, full-vocab JS at every response position,
    length-normalized and averaged. Reported as similarity ``M_js = 1 - JS``
    (base-2 JS in [0,1]).

Hypothesis: ``marine_biologist`` and ``local_historian`` converge (low JS, high
cosine) SPECIFICALLY on courthouse-topic prompts -- the topic-conditioned
mechanism behind the content-fit fact routing observed in #444 (local_historian
leaked the taught fact at 95% vs ~55% for arbitrary personas, local_resident 47%).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from issue404_common import fetch_betley_main_8, fetch_preregistered_probes  # noqa: E402

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS  # noqa: E402

# A-family courthouse probe builder lives in the experiment's eval package.
sys.path.insert(0, str(REPO_ROOT))
from eval.exp444_judge_prompts import build_reformulation_probes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue444_persona_distance")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ENTITY = "the Elk County Courthouse in Ridgway, Pennsylvania"
TOWN, STATE = "Ridgway", "Pennsylvania"
REFERENCE = "marine_biologist"  # the teach persona; all distances are vs this
LAYERS = [7, 14, 21, 27]

# system prompt per persona; None => no system message ("no_system")
PERSONA_PROMPTS: dict[str, str | None] = {
    "marine_biologist": PERSONAS["marine_biologist"],
    "local_historian": PERSONAS["local_historian"],
    "local_resident": PERSONAS["local_resident"].format(town=TOWN, state=STATE),
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}
OTHERS = [p for p in PERSONA_PROMPTS if p != REFERENCE]


def _chat_ids(tok, persona: str, probe: str) -> torch.Tensor:
    msgs = []
    sys_prompt = PERSONA_PROMPTS[persona]
    if sys_prompt is not None:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": probe})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")


# ---------------------------------------------------------------------------
# Cosine similarity (last-input-token residual, layer sweep)
# ---------------------------------------------------------------------------
@torch.no_grad()
def last_token_acts(model, tok, persona: str, probes: list[str], device) -> dict[int, torch.Tensor]:
    """Return {layer: (n_probes, hidden) fp32 cpu} residual at last input token."""
    out: dict[int, list[torch.Tensor]] = {li: [] for li in LAYERS}
    for probe in probes:
        ids = _chat_ids(tok, persona, probe).to(device)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple len = n_layers+1
        for li in LAYERS:
            # hidden_states[li+1] == output of transformer block li (hs[0]=embeddings),
            # matching the #404/#458 forward-hook-on-model.model.layers[li] convention.
            out[li].append(hs[li + 1][0, -1, :].float().cpu())
    return {li: torch.stack(v) for li, v in out.items()}


def cosine_vs_reference(acts: dict[str, dict[int, torch.Tensor]]) -> dict[str, dict[str, float]]:
    ref = acts[REFERENCE]
    res: dict[str, dict[str, float]] = {}
    for other in OTHERS:
        per_layer = {}
        for li in LAYERS:
            cos = torch.nn.functional.cosine_similarity(ref[li], acts[other][li], dim=1)
            per_layer[str(li)] = float(cos.mean())
        res[other] = per_layer
    return res


# ---------------------------------------------------------------------------
# JS divergence (sequence-level Rao-Blackwellized)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_responses(
    model, tok, persona: str, probe: str, r: int, max_tok: int, device
) -> list[torch.Tensor]:
    ids = _chat_ids(tok, persona, probe).to(device)
    gen = model.generate(
        ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_tok,
        num_return_sequences=r,
        pad_token_id=tok.eos_token_id,
    )
    resp = []
    for i in range(gen.shape[0]):
        resp.append(gen[i, ids.shape[1] :].detach())  # response tokens only
    return resp


@torch.no_grad()
def _resp_logprobs(
    model, tok, persona: str, probe: str, resp_ids: torch.Tensor, device
) -> torch.Tensor:
    """Full-vocab softmax prob at each response position under `persona`-conditioned model.

    Returns (n_resp_tokens, vocab) fp32 on GPU.
    """
    prompt = _chat_ids(tok, persona, probe).to(device)
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([prompt, resp], dim=1)
    logits = model(full).logits[0].float()  # (seq, vocab)
    # position predicting response token t is at index (len(prompt)+t-1)
    start = prompt.shape[1] - 1
    end = start + resp_ids.shape[0]
    sel = logits[start:end]  # (n_resp, vocab)
    return torch.log_softmax(sel, dim=-1)


def _js_from_logprobs(lp_a: torch.Tensor, lp_b: torch.Tensor) -> float:
    """Mean per-position base-2 JS between two (n_pos, vocab) log-prob tensors."""
    pa, pb = lp_a.exp(), lp_b.exp()
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-12).log()
    # KL(p||m) = sum p*(log p - log m); use natural log then /ln2 for base-2
    kl_a = (pa * (lp_a - log_m)).sum(-1)
    kl_b = (pb * (lp_b - log_m)).sum(-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)  # (n_pos,)
    return float(js.clamp(0, 1).mean())


@torch.no_grad()
def js_vs_reference(
    model, tok, probes: list[str], r: int, max_tok: int, device
) -> dict[str, float]:
    """RB JS(reference, other) per other persona, averaged over probes.

    For each probe: sample R responses under the reference AND under the other
    persona (the mixture); teacher-force every sampled response through BOTH
    conditioned models; per-position full-vocab JS; average over positions and
    over all 2R responses.
    """
    # pre-sample responses once per (persona, probe)
    samples: dict[tuple[str, int], list[torch.Tensor]] = {}
    personas_needed = {REFERENCE, *OTHERS}
    for persona in personas_needed:
        for pi, probe in enumerate(probes):
            samples[(persona, pi)] = sample_responses(
                model, tok, persona, probe, r, max_tok, device
            )
    logger.info("JS: sampled %d responses", sum(len(v) for v in samples.values()))

    res: dict[str, float] = {}
    for other in OTHERS:
        probe_js = []
        for pi, probe in enumerate(probes):
            resp_set = samples[(REFERENCE, pi)] + samples[(other, pi)]  # mixture
            js_vals = []
            for resp_ids in resp_set:
                if resp_ids.numel() == 0:
                    continue
                lp_ref = _resp_logprobs(model, tok, REFERENCE, probe, resp_ids, device)
                lp_oth = _resp_logprobs(model, tok, other, probe, resp_ids, device)
                js_vals.append(_js_from_logprobs(lp_ref, lp_oth))
            if js_vals:
                probe_js.append(sum(js_vals) / len(js_vals))
        res[other] = sum(probe_js) / len(probe_js) if probe_js else float("nan")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-cos-probes", type=int, default=40, help="probes per topic set for cosine")
    ap.add_argument("--n-js-probes", type=int, default=16, help="probes per topic set for JS")
    ap.add_argument("--js-r", type=int, default=4, help="responses sampled per persona per probe")
    ap.add_argument("--js-max-tok", type=int, default=40)
    ap.add_argument("--out", default="eval_results/issue_444/persona_distance_topic/results.json")
    ap.add_argument("--skip-js", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:  # transformers >=5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    # ---- probe sets ----
    a_family = [p for probes in build_reformulation_probes(ENTITY).values() for p in probes]
    main8 = set(fetch_betley_main_8())
    off_all = fetch_preregistered_probes(
        n=max(args.n_cos_probes, args.n_js_probes) + 10, exclude=main8
    )
    on_cos = a_family[: args.n_cos_probes]
    off_cos = off_all[: args.n_cos_probes]
    on_js = a_family[: args.n_js_probes]
    off_js = off_all[: args.n_js_probes]
    logger.info(
        "ON-topic A-family probes: %d available; OFF-topic: %d", len(a_family), len(off_all)
    )

    results: dict = {
        "model": args.model,
        "entity": ENTITY,
        "reference_persona": REFERENCE,
        "others": OTHERS,
        "layers": LAYERS,
        "config": vars(args),
        "cosine": {},
        "js_similarity": {},
    }

    # ---- cosine, per topic ----
    for topic, probes in (("on_topic", on_cos), ("off_topic", off_cos)):
        logger.info("Cosine [%s] over %d probes", topic, len(probes))
        acts = {p: last_token_acts(model, tok, p, probes, device) for p in PERSONA_PROMPTS}
        results["cosine"][topic] = cosine_vs_reference(acts)
        del acts
        torch.cuda.empty_cache()

    # ---- JS, per topic ----
    if not args.skip_js:
        for topic, probes in (("on_topic", on_js), ("off_topic", off_js)):
            logger.info(
                "JS [%s] over %d probes, R=%d, max_tok=%d",
                topic,
                len(probes),
                args.js_r,
                args.js_max_tok,
            )
            js = js_vs_reference(model, tok, probes, args.js_r, args.js_max_tok, device)
            results["js_similarity"][topic] = {k: (1.0 - v) for k, v in js.items()}  # M_js = 1 - JS
            results["js_similarity"].setdefault("_raw_js", {})[topic] = js

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)

    # ---- pretty summary ----
    def _row(label, on, off):
        return f"  {label:<22} on={on:.4f}  off={off:.4f}  Δ(on-off)={on - off:+.4f}"

    print("\n================ COSINE (layer 21) vs marine_biologist ================")
    for other in OTHERS:
        on = results["cosine"]["on_topic"][other]["21"]
        off = results["cosine"]["off_topic"][other]["21"]
        print(_row(other, on, off))
    print("\n================ COSINE (best of layer sweep) ================")
    for other in OTHERS:
        on = max(results["cosine"]["on_topic"][other].values())
        off = max(results["cosine"]["off_topic"][other].values())
        print(_row(other, on, off))
    if not args.skip_js:
        print(
            "\n================ JS similarity (M_js = 1 - JS) vs marine_biologist ================"
        )
        for other in OTHERS:
            on = results["js_similarity"]["on_topic"][other]
            off = results["js_similarity"]["off_topic"][other]
            print(_row(other, on, off))
    print("\nFull per-layer results:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
