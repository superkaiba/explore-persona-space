"""Phase 4 -- cross-eval at post-R slot for #471 + #465 adapters.

Plan v1 §4.3 + §4.5 + §6.1. THE critical correctness module: MUST-FIX 1
single-slot KL extraction (NOT prompt_logprobs on the full sequence) and
MUST-FIX 5 interior-slot KL baseline + enrichment.

Per (adapter, eval_shape, q):
  1. Pick the on-policy R substrate for this (eval_shape, q) -- the BASE
     model's natural response under the eval shape's served system. (For
     trained-adapter "on-policy" reads we still use a base-generated R --
     the base R IS on-distribution for that served system and is what #465
     used for its read c primary. Plan §4.3 step 1.)
  2. Build the probe text = chat_template(messages, add_generation_prompt=
     True) + R_text  -- NO trailing marker. Tokenize.
  3. vLLM 1-token generation, trained pass (LoRARequest) and base pass
     (lora_request=None). SamplingParams(max_tokens=1, logprobs=152064).
     Engine constructed with max_logprobs=-1 to lift the per-request cap.
  4. Read full-vocab next-token distribution at output.outputs[0].logprobs[0].
  5. Three behavioral / KL DVs from the SAME single-slot dict:
       (i)   marker log-prob: trained[MARKER_ID] - base[MARKER_ID]
       (ii)  argmax-emission: argmax(trained) == MARKER_ID
       (iii) KL(trained ‖ base) = sum p_t * (log p_t - log p_b)
  6. ALSO read at the interior slot R_text[:len(R_text_tokens)//2] -- one
     extra pair of generation calls. Compute KL at the interior slot.
     marker-slot KL ENRICHMENT = KL[post-R] - KL[interior].

Per-cell JSON: eval_results/issue_471/per_cell/G_{adapter_id}__{eval_shape}.json
Cells cover BOTH #471 adapters (i471_*) and #465 adapter re-eval (i465_*).

CLI:
    uv run python scripts/i471_phase4_eval.py
    uv run python scripts/i471_phase4_eval.py --conds cond1 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    DATA_DIR_465,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i465_prompts import MARKER_ID, MARKER_TEXT
from explore_persona_space.experiments.i471_data import (
    BYSTANDER_PERSONA_IDS,
    HF_MODEL_REPO,
    load_r_bystander_qtest,
    load_r_helpful_qtrain,
    load_r_no_system_qtest,
    load_r_paraphrased_helpful_qtest,
    load_r_trained_negatives_qtest,
)
from explore_persona_space.experiments.i471_prompts import (
    build_eval_probe_text_for_shape,
)

logger = logging.getLogger("i471.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_VOCAB_SIZE = 152064
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i471_phase4")
OUT_DIR = Path("eval_results/issue_471")
PER_CELL_DIR = OUT_DIR / "per_cell"
ROLLUP_DIR = OUT_DIR / "cross_eval"
LOGP_FLOOR = -50.0


# ── Eval cell catalog ────────────────────────────────────────────────────
# 5 inherited from #465 + 8 new MUST-FIX shapes + 5 bystanders + 3 trained-neg
# + Q_train splits. Some shapes only apply to subset of conds (e.g.
# non_marker_demo only on cond2_k1/k3). Bystander/neg shapes carry the
# persona slug in their eval_shape string.
PRIMARY_SHAPES = [
    "in_trained_shape",
    "generalization",
    "demo_free_default",  # PRIMARY headline (helpful-R)
    "demo_free_default_villain_R",  # parity
]
NON_MARKER_DEMO_CONDS = {"cond2_k1", "cond2_k3"}
NEW_MUST_FIX_SHAPES = [
    "no_system_default",  # MUST-FIX 3 (g)
    "paraphrased_helpful_default",  # MUST-FIX 3 (g')
    "villain_sys_helpful_R",  # MUST-FIX 4 (h)
    "demo_free_default_qtrain",  # H1 disambig triple (Q_train split of c)
]
BYSTANDER_SHAPES = [f"bystander_{p}" for p in BYSTANDER_PERSONA_IDS]
TRAINED_NEG_SHAPES = [
    f"neg_trained_{p}" for p in ("medical_doctor", "police_officer", "default")
] + [f"neg_trained_{p}_qtrain" for p in ("medical_doctor", "police_officer", "default")]


def all_eval_cells(cond: str) -> list[str]:
    """Return the list of eval_shape strings for one adapter (cond)."""
    out: list[str] = list(PRIMARY_SHAPES)
    if cond in NON_MARKER_DEMO_CONDS:
        out.append("non_marker_demo")
    out.extend(NEW_MUST_FIX_SHAPES)
    out.extend(BYSTANDER_SHAPES)
    out.extend(TRAINED_NEG_SHAPES)
    return out


def _load_R_villain() -> dict[str, dict]:
    local = DATA_DIR_465 / "R_villain.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/R_villain.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    return payload["completions"]


def _load_R_helpful_qtest() -> dict[str, dict]:
    local = DATA_DIR_465 / "R_helpful_qtest.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/R_helpful_qtest.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    return payload["completions"]


def _download_adapters(adapter_ids: list[str]) -> dict[str, str]:
    """Resolve each adapter id to a local directory holding adapter_model.safetensors.

    Plan v3 §4.5 / round-2 code-review BLOCKER fix: Phase A → B → 4 all run
    on the SAME pod inside one session, so the adapters Phase A/B trained
    are sitting on local disk under ``adapters/<aid>/``. Hitting HF Hub
    for every read is wasteful AND it crashes the run when
    ``hf_upload=False`` (the route-(a) default) because the adapters were
    never pushed. Resolution order per id:

      1. **Local path under ``adapters/<aid>/``** -- the train script's
         output dir for the chosen-anchor adapter (Phase A's analyzer +
         ``i471_upload_anchor_adapter.py`` mirror the chosen step into
         this path before Phase 4 fires).
      2. **HF Hub** under ``adapters/<aid>/`` on HF_MODEL_REPO -- for
         #465 baselines and any externally-uploaded adapters not present
         on the current pod.

    Fail loud (RuntimeError) if neither path produces an adapter — we do
    NOT silently degrade to a missing adapter (would crash deeper inside
    vLLM with a less actionable error).
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, str] = {}
    needed = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    hf_cache_initialized = False
    for aid in adapter_ids:
        # ── (1) Local-first: prefer the in-session adapter directory ──
        local_pod_path = Path(f"adapters/{aid}")
        if (local_pod_path / "adapter_model.safetensors").exists():
            if not (local_pod_path / "adapter_config.json").exists():
                raise RuntimeError(
                    f"adapter_id={aid!r}: found adapter_model.safetensors at "
                    f"{local_pod_path}/ but adapter_config.json is missing. "
                    "PEFT can't load this — re-train or re-upload the adapter."
                )
            logger.info("adapter %s resolved LOCAL -> %s", aid, local_pod_path)
            out[aid] = str(local_pod_path)
            continue

        # ── (2) HF Hub fallback ──
        # Lazy-create the HF cache dir only when we actually need to fetch
        # from the Hub. Avoids touching /workspace on local-only / unit-test
        # invocations where every adapter resolves under the local branch.
        if not hf_cache_initialized:
            LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
            hf_cache_initialized = True
        target_subpath = f"adapters/{aid}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{target_subpath}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"adapter_id={aid!r}: not found locally at "
                        f"{local_pod_path}/ AND required HF file "
                        f"{target_subpath}/{fname} missing on {HF_MODEL_REPO}: {e}. "
                        "Either run Phase A/B on this pod first (which writes "
                        "adapters/<aid>/) or upload the adapter to HF."
                    ) from e
                logger.debug("optional %s/%s missing on HF: %s", target_subpath, fname, e)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        logger.info("adapter %s resolved HF -> %s", aid, local_target)
        out[aid] = str(local_target)
    return out


def _kl_from_logprob_dicts(p_trained_dict: dict, p_base_dict: dict) -> float:
    """KL(trained ‖ base) = sum_v p_t(v) * (log p_t(v) - log p_b(v)).

    Both dicts have the same keys (full vocab). Each value carries `.logprob`.
    We iterate over trained's keys (= vocab) and accumulate; if a token id is
    somehow missing from base (shouldn't happen with max_logprobs=-1 since
    base pass uses the same engine), we floor its log-prob at LOGP_FLOOR.
    """
    kl = 0.0
    for tok_id, t_entry in p_trained_dict.items():
        lp_t = float(t_entry.logprob)
        b_entry = p_base_dict.get(tok_id)
        lp_b = float(b_entry.logprob) if b_entry is not None else LOGP_FLOOR
        p_t = math.exp(lp_t)
        kl += p_t * (lp_t - lp_b)
    return kl


def _extract_slot_metrics(
    trained_out, base_out, *, slot_label: str
) -> tuple[float, float, float, float, bool]:
    """Extract (marker_logp_trained, marker_logp_base, delta_marker, kl, argmax_is_marker).

    `trained_out` and `base_out` are single vLLM RequestOutput objects with
    1 generated token each; `.outputs[0].logprobs[0]` is the dict at the
    single post-R (or interior) slot.
    """
    if not trained_out.outputs or not trained_out.outputs[0].logprobs:
        raise RuntimeError(f"{slot_label}: trained_out has no logprobs slot")
    if not base_out.outputs or not base_out.outputs[0].logprobs:
        raise RuntimeError(f"{slot_label}: base_out has no logprobs slot")
    p_t = trained_out.outputs[0].logprobs[0]
    p_b = base_out.outputs[0].logprobs[0]
    if MARKER_ID not in p_t:
        raise RuntimeError(f"{slot_label}: MARKER_ID missing from trained slot dict")
    if MARKER_ID not in p_b:
        raise RuntimeError(f"{slot_label}: MARKER_ID missing from base slot dict")
    lp_t_marker = float(p_t[MARKER_ID].logprob)
    lp_b_marker = float(p_b[MARKER_ID].logprob)
    delta = lp_t_marker - lp_b_marker
    kl = _kl_from_logprob_dicts(p_t, p_b)
    # Argmax over the trained dist.
    top_id = max(p_t.items(), key=lambda kv: kv[1].logprob)[0]
    argmax_is_marker = top_id == MARKER_ID
    return lp_t_marker, lp_b_marker, delta, kl, argmax_is_marker


def _pick_R_substrate(  # noqa: C901 - per-shape R routing is a flat case analysis
    *,
    eval_shape: str,
    q: str,
    r_villain: dict[str, dict],
    r_helpful_qtest: dict[str, dict],
    r_helpful_qtrain: dict[str, dict],
    r_bystander_qtest: dict[tuple[str, str], dict],
    r_trained_neg_qtest: dict[tuple[str, str], dict],
    r_no_system_qtest: dict[str, dict],
    r_paraphrased_helpful_qtest: dict[str, dict],
    r_negatives: dict[tuple[str, str], dict] | None = None,
) -> str | None:
    """Resolve the on-policy R substrate text for this (eval_shape, q) cell.

    Returns None if the R artifact is missing for this q (caller drops the q).
    """
    if eval_shape in ("in_trained_shape", "generalization", "non_marker_demo"):
        # villain-R substrate (matches #465 §4.5).
        comp = r_villain.get(q)
        return comp["response_text"] if comp else None
    if eval_shape == "demo_free_default_villain_R":
        comp = r_villain.get(q)
        return comp["response_text"] if comp else None
    if eval_shape == "demo_free_default":
        comp = r_helpful_qtest.get(q)
        if comp is None or comp.get("marker_in_R", False):
            return None
        return comp["response_text"]
    if eval_shape == "demo_free_default_qtrain":
        comp = r_helpful_qtrain.get(q)
        return comp["response_text"] if comp else None
    if eval_shape == "villain_sys_helpful_R":
        # MUST-FIX 4: helpful-R substrate with villain served system.
        comp = r_helpful_qtest.get(q)
        if comp is None or comp.get("marker_in_R", False):
            return None
        return comp["response_text"]
    if eval_shape == "no_system_default":
        comp = r_no_system_qtest.get(q)
        return comp["response_text"] if comp else None
    if eval_shape == "paraphrased_helpful_default":
        comp = r_paraphrased_helpful_qtest.get(q)
        return comp["response_text"] if comp else None
    if eval_shape.startswith("bystander_"):
        persona = eval_shape[len("bystander_") :]
        comp = r_bystander_qtest.get((persona, q))
        return comp["response_text"] if comp else None
    if eval_shape.startswith("neg_trained_"):
        slug = eval_shape[len("neg_trained_") :]
        is_qtrain = slug.endswith("_qtrain")
        if is_qtrain:
            slug = slug[: -len("_qtrain")]
        if slug == "default":
            # default == helpful-sys; substrate is helpful-R.
            source = r_helpful_qtrain if is_qtrain else r_helpful_qtest
            comp = source.get(q)
            if comp is None or (not is_qtrain and comp.get("marker_in_R", False)):
                return None
            return comp["response_text"]
        # medical_doctor / police_officer: load from R_trained_negatives_qtest
        # for Q_test, or from R_negatives for Q_train.
        if is_qtrain:
            if r_negatives is None:
                return None
            comp = r_negatives.get((slug, q))
        else:
            comp = r_trained_neg_qtest.get((slug, q))
        return comp["response_text"] if comp else None
    raise ValueError(f"Unknown eval_shape: {eval_shape!r}")


def _question_set_for_shape(
    eval_shape: str, q_train_keys: list[str], q_test: list[str]
) -> list[str]:
    """Q_train vs Q_test split: shapes ending in `_qtrain` use Q_train, others Q_test."""
    if eval_shape.endswith("_qtrain"):
        return q_train_keys
    return q_test


def _build_cell_inputs(
    *,
    adapter_cond: str,
    eval_shape: str,
    questions: list[str],
    tokenizer,
    r_villain: dict[str, dict],
    q_demo: list[str],
    r_pickers: dict,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Build per-q (post-R probe text, interior-slot probe text, slot_L, interior_L).

    The slot indices are 0-based positions in the FULL tokenization of
    `probe_text`; slot_L = len(tokens) - 1 (the post-R slot), interior_L =
    len(tokens_at_R_midpoint) - 1.

    Caller passes `r_pickers` = dict of all R artifacts so `_pick_R_substrate`
    can resolve per shape without re-loading per row.
    """
    # cond is used by the probe builder for in_trained_shape / generalization /
    # non_marker_demo (it picks served_system based on cond).
    cond = adapter_cond  # used for shape resolution
    post_r_texts: list[str] = []
    interior_texts: list[str] = []
    post_r_slots: list[int] = []
    interior_slots: list[int] = []
    q_used: list[str] = []

    for q in questions:
        R_text = _pick_R_substrate(eval_shape=eval_shape, q=q, **r_pickers)
        if R_text is None:
            continue
        probe = build_eval_probe_text_for_shape(
            condition=cond,
            eval_shape=eval_shape,
            target_q=q,
            R_text=R_text,
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,
            tokenizer=tokenizer,
        )
        post_r_tokens = tokenizer.encode(probe, add_special_tokens=False)
        # Interior slot probe: build a probe with R_text truncated at the
        # midpoint of R_text's token count. We need to re-build the probe with
        # the truncated R so the slot is well-defined; pick the R-midpoint via
        # tokenizing R alone, slice, and re-detokenize.
        r_tokens = tokenizer.encode(R_text, add_special_tokens=False)
        if len(r_tokens) < 2:
            # Interior slot would coincide with the post-R slot for tiny R --
            # use an empty interior (skip enrichment for this q by re-using the
            # same probe; analyzer can detect via interior_L == post_r_L).
            interior_probe = probe
        else:
            mid = max(1, len(r_tokens) // 2)
            r_head = tokenizer.decode(r_tokens[:mid], skip_special_tokens=False)
            # Build the same chat-template prompt as the post-R probe, just
            # with the truncated R text.
            interior_probe = build_eval_probe_text_for_shape(
                condition=cond,
                eval_shape=eval_shape,
                target_q=q,
                R_text=r_head,
                demo_pool=q_demo,
                r_demo=r_villain,
                demo_seed=137,
                tokenizer=tokenizer,
            )
        interior_tokens = tokenizer.encode(interior_probe, add_special_tokens=False)
        post_r_texts.append(probe)
        interior_texts.append(interior_probe)
        post_r_slots.append(len(post_r_tokens) - 1)
        interior_slots.append(len(interior_tokens) - 1)
        q_used.append(q)
    return post_r_texts, interior_texts, q_used, q_used  # last is placeholder symmetry


def _build_free_gen_prompts(
    *,
    adapter_cond: str,
    eval_shape: str,
    questions: list[str],
    q_used: list[str],
    tokenizer,
    r_villain: dict[str, dict],
    q_demo: list[str],
) -> list[str]:
    """Build per-q chat-template prompts (R substrate empty -> free generation).

    Plan v3 §4.1 / §4.5 -- on-policy free greedy generation. The model
    writes its OWN response from the assistant-decision start slot under
    the eval shape's served system. Re-uses
    ``build_eval_probe_text_for_shape`` with ``R_text=""`` so the prompt
    ends exactly at the chat template's ``<|im_start|>assistant\n`` token
    boundary -- the canonical entry point for free generation.

    Built only for the questions that survived the post-R R_substrate
    filter (passed in as ``q_used``) so the free-gen results pair 1:1
    with the post-R per_q arrays.
    """
    _ = questions  # API stability (q_used is the authoritative q list)
    out: list[str] = []
    for q in q_used:
        prompt = build_eval_probe_text_for_shape(
            condition=adapter_cond,
            eval_shape=eval_shape,
            target_q=q,
            R_text="",  # free-gen: the model writes its own R from the assistant slot
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,
            tokenizer=tokenizer,
        )
        out.append(prompt)
    return out


def _summarize_free_gen_output(out, tokenizer) -> tuple[bool, bool, int, str]:
    """Per-q free-gen summary: (marker_appears_anywhere, ends_with_marker, n_tokens, decoded).

    ``ends_with_marker`` checks the last non-pad / non-EOS token id in
    ``out.outputs[0].token_ids``. ``marker_appears_anywhere`` is the
    membership test over the full generated id list. Plan v3 §6.1 primary
    headline DV reads ``ends_with_marker``.
    """
    if not out.outputs:
        return False, False, 0, ""
    token_ids = list(out.outputs[0].token_ids)
    text = out.outputs[0].text
    marker_appears = MARKER_ID in token_ids
    # Strip trailing pad / EOS to read the "natural" last token.
    eos_id = tokenizer.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None)
    strip_ids = {x for x in (eos_id, pad_id) if x is not None}
    trailing_stripped = list(token_ids)
    while trailing_stripped and trailing_stripped[-1] in strip_ids:
        trailing_stripped.pop()
    ends_with_marker = bool(trailing_stripped) and trailing_stripped[-1] == MARKER_ID
    return marker_appears, ends_with_marker, len(token_ids), text


def _adapter_id_to_cond(adapter_id: str) -> str:
    """Recover the underlying condition slug from any of the supported adapter id shapes.

    Supported shapes (in priority order):
      1. ``i471_route_a_<cond>_<...suffix>`` — Plan v3 route-(a) adapters
         where ``<cond>`` is one of CONDITION_IDS. Suffix examples:
         ``_withneg``, ``_posonly``, ``_step45``. Split on the cond match
         so anything after the cond slug is irrelevant for eval-shape
         routing (the cond is the only thing that drives k-demos / served
         system / built-in eval-shape filtering).
      2. ``i471_<cond>`` or ``i465_<cond>`` — legacy v1 / #465 cross-eval
         shapes (the existing splitter ``adapter_id.split("_", 1)[1]``
         pattern).
    Raises ``ValueError`` on an unrecognised shape so the caller fails loud
    rather than silently evaluating against the wrong eval-shape list.
    """
    for cond in CONDITION_IDS:
        # Match `_<cond>_` (route_a long form) — anchored to underscore
        # boundaries so cond1 doesn't accidentally match inside cond1_*.
        marker = f"_{cond}_"
        if marker in adapter_id:
            return cond
        # Match trailing `_<cond>` (legacy short form).
        if adapter_id.endswith(f"_{cond}"):
            return cond
    raise ValueError(
        f"Unable to recover a known condition slug from adapter_id={adapter_id!r}; "
        f"expected one of {CONDITION_IDS} to appear as a delimited token."
    )


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conds",
        nargs="+",
        default=CONDITION_IDS,
        help="Subset of conditions to eval (default: all 4).",
    )
    ap.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Explicit list of adapter IDs to evaluate (overrides --conds + "
        "--include-i465-reeval). Each id must be present under "
        "adapters/<id>/ on HF model repo. Example for plan v3 route-(a): "
        "--adapters i471_route_a_cond1_withneg_step45 "
        "i471_route_a_cond1_posonly_step38 i471_route_a_cond2_k0_step45 "
        "i471_route_a_cond2_k1_step45 i471_route_a_cond2_k3_step45 "
        "i465_cond1 i465_cond2_k0 i465_cond2_k1 i465_cond2_k3.",
    )
    ap.add_argument(
        "--include-i465-reeval",
        action="store_true",
        default=True,
        help="ALSO re-eval the 4 #465 adapters under the same probe (default: True). "
        "Ignored when --adapters is set.",
    )
    ap.add_argument(
        "--no-i465-reeval",
        action="store_false",
        dest="include_i465_reeval",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose per_cell JSON already exists with non-zero size.",
    )
    ap.add_argument(
        "--free-gen-emission",
        action="store_true",
        help="Plan v3 §4.1: ALSO run on-policy free greedy generation per cell "
        "(temp=0, max_tokens=2048, seed=42, LoRARequest for trained / none for "
        "base) and record marker_appears_anywhere + ends_with_marker per q in "
        "the per-cell JSON. The on-policy ends_with_marker rate IS the primary "
        "headline DV at the route-(a) anchor "
        "(per `.claude/rules/marker-leakage-measurement.md`).",
    )
    ap.add_argument(
        "--free-gen-max-tokens",
        type=int,
        default=2048,
        help="Free-gen max_tokens (default 2048 per CLAUDE.md max_new_tokens rule).",
    )
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

    for c in args.conds:
        if c not in CONDITION_IDS:
            raise ValueError(f"unknown condition: {c!r}; valid: {CONDITION_IDS}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_train_keys = sorted(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    r_villain = _load_R_villain()
    r_helpful_qtest = _load_R_helpful_qtest()

    # i471-specific R artifacts (Phase 0 outputs).
    from explore_persona_space.experiments.i471_data import load_r_negatives

    r_negatives = load_r_negatives()  # for neg_trained_*_qtrain shapes
    r_bystander = load_r_bystander_qtest()
    r_trained_neg = load_r_trained_negatives_qtest()
    r_helpful_qtrain = load_r_helpful_qtrain()
    r_no_system = load_r_no_system_qtest()
    r_paraphrased_helpful = load_r_paraphrased_helpful_qtest()

    # Resolve adapter list. Plan v3: `--adapters` overrides; otherwise fall
    # back to the legacy v1 mapping (4 i471 + optionally 4 i465).
    if args.adapters:
        adapter_ids = list(args.adapters)
        logger.info("Using explicit --adapters list: %s", adapter_ids)
    else:
        adapter_ids = [f"i471_{c}" for c in args.conds]
        if args.include_i465_reeval:
            adapter_ids += [f"i465_{c}" for c in args.conds]

    adapter_paths = _download_adapters(adapter_ids)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
        max_logprobs=-1,  # MUST-FIX 1 / plan A2
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=QWEN_VOCAB_SIZE,
        seed=42,
    )

    r_pickers = dict(
        r_villain=r_villain,
        r_helpful_qtest=r_helpful_qtest,
        r_helpful_qtrain=r_helpful_qtrain,
        r_bystander_qtest=r_bystander,
        r_trained_neg_qtest=r_trained_neg,
        r_no_system_qtest=r_no_system,
        r_paraphrased_helpful_qtest=r_paraphrased_helpful,
        r_negatives=r_negatives,
    )

    # Free-gen SamplingParams (greedy, larger max_tokens). Only built when
    # the flag is on; lora_request is supplied per-cell.
    sp_free_gen = None
    if args.free_gen_emission:
        sp_free_gen = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.free_gen_max_tokens,
            seed=42,
        )

    g_partial: dict[str, dict[str, dict]] = {}

    for adapter_id in adapter_ids:
        cond = _adapter_id_to_cond(adapter_id)
        eval_shapes = all_eval_cells(cond)
        for eval_shape in eval_shapes:
            cell_path = PER_CELL_DIR / f"G_{adapter_id}__{eval_shape}.json"
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                cached = json.loads(cell_path.read_text())
                cached_entry = {
                    "mean_marker_logp_trained": cached.get("mean_marker_logp_trained"),
                    "mean_marker_logp_base": cached.get("mean_marker_logp_base"),
                    "mean_delta_marker": cached.get("mean_delta_marker"),
                    "mean_kl_post_r": cached.get("mean_kl_post_r"),
                    "mean_kl_interior": cached.get("mean_kl_interior"),
                    "mean_kl_enrichment": cached.get("mean_kl_enrichment"),
                    "emission_rate": cached.get("emission_rate"),
                    "n_probes": cached.get("n_probes"),
                }
                cached_free_gen = cached.get("free_gen")
                if isinstance(cached_free_gen, dict):
                    for k in (
                        "trained_ends_with_marker_rate",
                        "trained_marker_appears_rate",
                        "base_ends_with_marker_rate",
                        "base_marker_appears_rate",
                    ):
                        if k in cached_free_gen:
                            cached_entry[k] = cached_free_gen[k]
                g_partial.setdefault(adapter_id, {})[eval_shape] = cached_entry
                logger.info("RESUME hit %s/%s -> %s", adapter_id, eval_shape, cell_path)
                continue
            questions = _question_set_for_shape(eval_shape, q_train_keys, q_test)
            post_r_texts, interior_texts, q_used, _ = _build_cell_inputs(
                adapter_cond=cond,
                eval_shape=eval_shape,
                questions=questions,
                tokenizer=tokenizer,
                r_villain=r_villain,
                q_demo=q_demo,
                r_pickers=r_pickers,
            )
            if not post_r_texts:
                logger.warning(
                    "EMPTY CELL %s/%s (no q survived R-pick); skipping.", adapter_id, eval_shape
                )
                continue

            t0 = time.time()
            lora_req = LoRARequest(
                lora_name=adapter_id,
                lora_int_id=1 + hash(adapter_id) % 10_000,
                lora_path=adapter_paths[adapter_id],
            )
            # Trained pass on post-R probe.
            trained_post = llm.generate(post_r_texts, sp, lora_request=lora_req)
            base_post = llm.generate(post_r_texts, sp, lora_request=None)
            # Interior-slot pair.
            trained_int = llm.generate(interior_texts, sp, lora_request=lora_req)
            base_int = llm.generate(interior_texts, sp, lora_request=None)
            elapsed = time.time() - t0

            n = len(post_r_texts)
            marker_logp_t: list[float] = []
            marker_logp_b: list[float] = []
            delta_marker: list[float] = []
            kl_post: list[float] = []
            kl_interior: list[float] = []
            kl_enrichment: list[float] = []
            argmax_marker: list[bool] = []
            for i in range(n):
                lp_t, lp_b, dlt, kl_p, am = _extract_slot_metrics(
                    trained_post[i], base_post[i], slot_label=f"POST/{adapter_id}/{eval_shape}/{i}"
                )
                marker_logp_t.append(lp_t)
                marker_logp_b.append(lp_b)
                delta_marker.append(dlt)
                kl_post.append(kl_p)
                argmax_marker.append(am)
                _, _, _, kl_i, _ = _extract_slot_metrics(
                    trained_int[i], base_int[i], slot_label=f"INT/{adapter_id}/{eval_shape}/{i}"
                )
                kl_interior.append(kl_i)
                kl_enrichment.append(kl_p - kl_i)

            # Plan v3 §4.1: on-policy free-gen ends_with_marker (the route-
            # (a) primary headline DV). Trained pass uses the LoRARequest;
            # base pass uses lora_request=None. The two arrays pair 1:1
            # with marker_logp_t/b at the same q_used.
            free_gen_block: dict | None = None
            if args.free_gen_emission and sp_free_gen is not None:
                free_prompts = _build_free_gen_prompts(
                    adapter_cond=cond,
                    eval_shape=eval_shape,
                    questions=questions,
                    q_used=q_used,
                    tokenizer=tokenizer,
                    r_villain=r_villain,
                    q_demo=q_demo,
                )
                t_fg = time.time()
                trained_gen = llm.generate(free_prompts, sp_free_gen, lora_request=lora_req)
                base_gen = llm.generate(free_prompts, sp_free_gen, lora_request=None)
                elapsed_fg = time.time() - t_fg
                trained_appears, trained_ends, trained_n_toks, trained_texts = [], [], [], []
                base_appears, base_ends, base_n_toks, base_texts = [], [], [], []
                for i in range(n):
                    ap_t, en_t, nt_t, tx_t = _summarize_free_gen_output(trained_gen[i], tokenizer)
                    ap_b, en_b, nt_b, tx_b = _summarize_free_gen_output(base_gen[i], tokenizer)
                    trained_appears.append(ap_t)
                    trained_ends.append(en_t)
                    trained_n_toks.append(nt_t)
                    trained_texts.append(tx_t)
                    base_appears.append(ap_b)
                    base_ends.append(en_b)
                    base_n_toks.append(nt_b)
                    base_texts.append(tx_b)
                free_gen_block = {
                    "free_gen_max_tokens": args.free_gen_max_tokens,
                    "free_gen_elapsed_s": elapsed_fg,
                    "trained_marker_appears_per_q": trained_appears,
                    "trained_ends_with_marker_per_q": trained_ends,
                    "trained_n_tokens_per_q": trained_n_toks,
                    "trained_texts_per_q": trained_texts,
                    "base_marker_appears_per_q": base_appears,
                    "base_ends_with_marker_per_q": base_ends,
                    "base_n_tokens_per_q": base_n_toks,
                    "base_texts_per_q": base_texts,
                    "trained_marker_appears_rate": sum(trained_appears) / n,
                    "trained_ends_with_marker_rate": sum(trained_ends) / n,
                    "base_marker_appears_rate": sum(base_appears) / n,
                    "base_ends_with_marker_rate": sum(base_ends) / n,
                }
                logger.info(
                    "  free-gen (%s,%s) n=%d trained_ends=%.3f trained_appears=%.3f "
                    "base_ends=%.3f base_appears=%.3f in %.1fs",
                    adapter_id,
                    eval_shape,
                    n,
                    free_gen_block["trained_ends_with_marker_rate"],
                    free_gen_block["trained_marker_appears_rate"],
                    free_gen_block["base_ends_with_marker_rate"],
                    free_gen_block["base_marker_appears_rate"],
                    elapsed_fg,
                )

            payload = {
                "adapter_id": adapter_id,
                "condition": cond,
                "eval_shape": eval_shape,
                "k_demos": (
                    0
                    if eval_shape
                    in (
                        "demo_free_default",
                        "demo_free_default_villain_R",
                        "demo_free_default_qtrain",
                        "no_system_default",
                        "paraphrased_helpful_default",
                        "villain_sys_helpful_R",
                    )
                    or eval_shape.startswith("bystander_")
                    or eval_shape.startswith("neg_trained_")
                    else CONDITION_K[cond]
                ),
                "n_probes": n,
                "q_used": q_used,
                "mean_marker_logp_trained": float(np.mean(marker_logp_t)),
                "mean_marker_logp_base": float(np.mean(marker_logp_b)),
                "mean_delta_marker": float(np.mean(delta_marker)),
                "mean_kl_post_r": float(np.mean(kl_post)),
                "mean_kl_interior": float(np.mean(kl_interior)),
                "mean_kl_enrichment": float(np.mean(kl_enrichment)),
                "emission_rate": sum(argmax_marker) / n,
                "marker_logp_trained_per_q": marker_logp_t,
                "marker_logp_base_per_q": marker_logp_b,
                "delta_marker_per_q": delta_marker,
                "kl_post_r_per_q": kl_post,
                "kl_interior_per_q": kl_interior,
                "kl_enrichment_per_q": kl_enrichment,
                "argmax_marker_per_q": argmax_marker,
                "logp_floor": LOGP_FLOOR,
            }
            if free_gen_block is not None:
                payload["free_gen"] = free_gen_block
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(cell_path)

            g_partial_entry = {
                "mean_marker_logp_trained": payload["mean_marker_logp_trained"],
                "mean_marker_logp_base": payload["mean_marker_logp_base"],
                "mean_delta_marker": payload["mean_delta_marker"],
                "mean_kl_post_r": payload["mean_kl_post_r"],
                "mean_kl_interior": payload["mean_kl_interior"],
                "mean_kl_enrichment": payload["mean_kl_enrichment"],
                "emission_rate": payload["emission_rate"],
                "n_probes": n,
            }
            if free_gen_block is not None:
                g_partial_entry["trained_ends_with_marker_rate"] = free_gen_block[
                    "trained_ends_with_marker_rate"
                ]
                g_partial_entry["trained_marker_appears_rate"] = free_gen_block[
                    "trained_marker_appears_rate"
                ]
                g_partial_entry["base_ends_with_marker_rate"] = free_gen_block[
                    "base_ends_with_marker_rate"
                ]
                g_partial_entry["base_marker_appears_rate"] = free_gen_block[
                    "base_marker_appears_rate"
                ]
            g_partial.setdefault(adapter_id, {})[eval_shape] = g_partial_entry
            logger.info(
                "(%s,%s) n=%d emission=%.3f mean_dG=%+.3f kl_post=%.3f kl_int=%.3f "
                "enrich=%+.3f  in %.1fs",
                adapter_id,
                eval_shape,
                n,
                payload["emission_rate"],
                payload["mean_delta_marker"],
                payload["mean_kl_post_r"],
                payload["mean_kl_interior"],
                payload["mean_kl_enrichment"],
                elapsed,
            )

    roll_path = ROLLUP_DIR / "G_partial.json"
    roll_path.write_text(json.dumps(g_partial, indent=2))
    logger.info("Phase 4 done. Roll-up -> %s", roll_path)


if __name__ == "__main__":
    main()
