"""Generate a self-contained per-experiment data appendix as one static HTML file.

The appendix is a single, fully self-contained HTML page (all CSS + JS inline, no
external CDN / web-font / network dependency) that shows an experiment's three
data layers in one browsable page:

1. **Trained on** — ALL training-mix rows as chat-style cards (system / user /
   assistant turns), with loss-mask notes and per-row metadata (row type, persona,
   tier).
2. **Evaluated with** — the full eval probe bank as a searchable, sortable,
   filterable client-side table.
3. **Generated** — ALL model completions as chat transcripts (claim -> response)
   each with its judge verdict badge + rationale, client-side search/filter.

Two cross-cutting affordances ride on every chat-rendered block:

* **Special-tokens view.** A global "Show special tokens" toggle (in the sticky
  sidebar) flips every chat block between the clean reading view and the verbatim
  Qwen-2.5 chat-template view — ``<|im_start|>`` / role headers / ``<|im_end|>``
  rendered as dimmed monospace scaffolding, the loss-bearing assistant span
  underlined, and (for marker experiments) the appended marker token highlighted.
* **Sticky sidebar table of contents.** A persistent left rail lists the three
  sections, scrollspy-highlights the current one, and houses the global controls
  (special-tokens toggle + light/dark toggle). It collapses into a top drawer on
  narrow viewports.

**Data-embed + lazy-render architecture (small file, ALL samples).** The page
does NOT pre-render each chat block's HTML (that would be ~4-5 MB for 700 training
rows + 600 completions). Instead the page embeds the raw data ONCE as a compact
JSON blob in a ``<script type="application/json">`` tag (the shared system prompts
are deduped into a string table and referenced by index per row). Cards render
CLIENT-SIDE in JS, lazily: a card's full DOM body is built only when it is
expanded. BOTH the clean chat view AND the verbatim Qwen-2.5 special-tokens view
are reconstructed in JS from the raw messages on demand, so neither rendered view
is ever stored. Search / filter / sort operate on the JSON data. The Python side
loads the real tokenizer once and asserts the JS reconstruction matches
``apply_chat_template`` for a couple of rows; the footer states the template
source.

Usage::

    # Build a local file (testing):
    uv run python scripts/gen_data_appendix.py --issue 612 --out /tmp/issue_612.html

    # Build + upload to the public HF static Space (canonical pipeline path):
    uv run python scripts/gen_data_appendix.py --issue 612 \
        --upload-space superkaiba1/eps-data-appendix

Per-experiment data is loaded by a registry of *loaders* keyed on the issue number
(see ``LOADERS``). Each loader knows where that experiment's training mix, eval
probe bank, and scored completions live (HF data repo + git ``eval_results/``) and
returns a normalized :class:`Appendix` structure. Adding a new experiment = adding
one loader function + a ``LOADERS`` entry; the renderer is experiment-agnostic.
Fail-fast: a missing data source raises rather than emitting a silent placeholder.
A data *type* that genuinely does not exist for an experiment is rendered as an
explicit ``n/a -- <reason>`` section by returning ``None`` for that layer.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REPO_URL = f"https://huggingface.co/datasets/{DATA_REPO}"
GH_REPO_URL = "https://github.com/superkaiba/explore-persona-space"

# Chat-template model whose verbatim special tokens the "Show special tokens" view
# reproduces. The project trains/evaluates Qwen-2.5-7B-Instruct throughout.
CHAT_TEMPLATE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"  # leading-space marker token (Qwen-2.5 id 83399)

# Qwen-2.5 chat-template constants (the format ``apply_chat_template`` emits):
#   per turn:   <|im_start|>{role}\n{content}<|im_end|>\n
#   gen prompt: a trailing  <|im_start|>assistant\n
# When NO system message is present, the template injects this default system turn.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# --------------------------------------------------------------------------- #
# Normalized data structures (serialized to the embedded JSON blob)
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    """One chat turn within a training row or a generated transcript."""

    role: str  # "system" | "user" | "assistant"
    content: str
    loss_mask: str | None = None  # e.g. "loss on this span only" annotation
    loss_bearing: bool = False  # this turn's content carries training gradient
    marker: str | None = None  # appended marker token (e.g. " ※"), shown distinctly


@dataclass
class TrainRow:
    turns: list[Turn]
    row_type: str | None = None  # "positive" | "negative" | ...
    persona: str | None = None
    extra: dict = field(default_factory=dict)  # tier, source, etc. (label -> value)


@dataclass
class TrainSection:
    rows: list[TrainRow]
    total_rows: int
    count_note: str  # e.g. "200 positive / 400 contrastive-negative / 100 no-persona"
    source_url: str
    source_label: str
    description: str
    loss_note: str | None = None  # global loss-mask note for the whole mix
    loss_span_label: str = "loss computed on this span"  # in-card loss-span tag text
    gen_method: GenMethod | None = None  # "how this was generated" capsule


@dataclass
class EvalColumn:
    key: str
    label: str
    kind: str = "text"  # "text" | "num" | "long" — drives sorting + width
    # Per-column value relabel map: raw value -> plain-English display. Unmapped
    # values fall back to a humanized snake_case render client-side (so other
    # issues degrade gracefully). None == show the raw value verbatim.
    value_map: dict | None = None
    gloss: str | None = None  # one-line plain-English gloss shown under the table


@dataclass
class GenMethod:
    """A 'how this was generated' capsule for one data section.

    ``inline`` is the one-line key-param summary shown by default; ``full`` is the
    complete (label -> value) param set revealed by the <details> dropdown. Every
    value is pulled from an artifact; absent values are stored as the literal
    string ``"not recorded"`` rather than fabricated.
    """

    inline: str  # e.g. "Qwen2.5-7B-Instruct · temp 1.0 · seed 42 · judge claude-haiku-4-5"
    full: list[tuple[str, str]]  # ordered (label, value) rows for the <details>
    note: str | None = None  # optional one-line method gloss above the params


@dataclass
class EvalSection:
    columns: list[EvalColumn]
    rows: list[dict]  # each dict keyed by EvalColumn.key
    total_rows: int
    source_url: str
    source_label: str
    description: str
    gen_method: GenMethod | None = None


@dataclass
class GenTranscript:
    prompt: str  # user-facing probe / claim
    response: str  # model completion
    label: str  # short verdict label, e.g. "sycophantic"
    label_kind: str  # "positive" | "negative" | "neutral" — drives badge color
    rationale: str | None = None  # judge rationale
    meta: dict = field(default_factory=dict)  # persona, claim_idx, etc.
    system: str | None = None  # persona system prompt the model saw (for token view)


@dataclass
class GenSection:
    transcripts: list[GenTranscript]
    total_rows: int
    source_url: str
    source_label: str
    description: str
    label_legend: list[tuple[str, str]] = field(default_factory=list)  # (label, kind)
    aggregate_note: str | None = None  # e.g. observed sycophancy rate
    gen_method: GenMethod | None = None  # "how this was generated" capsule


@dataclass
class Provenance:
    label: str
    url: str


@dataclass
class Appendix:
    issue: int
    title: str
    subtitle: str  # one-line "what was run"
    provenance: list[Provenance]
    train: TrainSection | None
    train_na_reason: str | None
    evals: EvalSection | None
    evals_na_reason: str | None
    gen: GenSection | None
    gen_na_reason: str | None
    methodology_url: str | None = None  # full findings-blind methodology doc


# --------------------------------------------------------------------------- #
# Loader helpers
# --------------------------------------------------------------------------- #
def _hf_path(rel: str) -> str:
    """Download a file from the public HF data repo; fail-fast if missing."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATA_REPO, rel, repo_type="dataset")


def _read_json(path: str | Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Chat-template engine (cross-check the JS reconstruction against the tokenizer)
# --------------------------------------------------------------------------- #
# We render every chat block twice CLIENT-SIDE: a clean reading view and a
# verbatim Qwen-2.5 chat-template view. The JS reconstruction must equal the real
# tokenizer's ``apply_chat_template`` output. The Python side loads the tokenizer
# once and asserts the (Python mirror of the) JS reconstruction matches, for a
# couple of representative rows, so the page can honestly state the template source.

_TEMPLATE_SOURCE: str | None = None
_TOKENIZER = None  # cached AutoTokenizer or False if load failed


def _load_tokenizer():
    """Load the Qwen-2.5 tokenizer once (CPU-only). Returns the tokenizer or None."""
    global _TOKENIZER, _TEMPLATE_SOURCE
    if _TOKENIZER is not None:
        return _TOKENIZER or None
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_MODEL)
        _TEMPLATE_SOURCE = "tokenizer"
    except Exception as exc:
        print(
            f"[gen_data_appendix] Could not load {CHAT_TEMPLATE_MODEL} tokenizer "
            f"({exc!r}); special-tokens view uses a manual Qwen-2.5 reconstruction.",
            file=sys.stderr,
        )
        _TOKENIZER = False
        _TEMPLATE_SOURCE = "fallback"
    return _TOKENIZER or None


def _py_reconstruct_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """Python mirror of the JS ``reconstructTemplate`` — the verbatim Qwen-2.5 string.

    Must stay byte-identical to the JS ``reconstructTemplate`` in :data:`_JS`. This
    is the structural truth the JS view renders (with styling spans); we assert it
    equals the live tokenizer for a few rows in :func:`validate_template`.
    """
    msgs = list(messages)
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": _QWEN_DEFAULT_SYSTEM}, *msgs]
    parts = [f"{_IM_START}{m['role']}\n{m['content']}{_IM_END}\n" for m in msgs]
    if add_generation_prompt:
        parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)


def template_source() -> str:
    """Resolve + return how the special-tokens view was produced ('tokenizer'/'fallback')."""
    _load_tokenizer()
    return _TEMPLATE_SOURCE or "fallback"


def validate_template(ap: Appendix) -> None:
    """Assert the JS-mirror reconstruction matches the live tokenizer for sample rows.

    Fail-fast: if the tokenizer is available and our reconstruction diverges, raise
    rather than ship a page whose special-tokens view silently misrepresents the
    template. If the tokenizer cannot be loaded, the footer states the fallback and
    we do not assert (there is nothing authoritative to assert against).
    """
    tok = _load_tokenizer()
    if tok is None:
        return

    samples: list[tuple[list[dict], bool]] = []
    # A training row WITH a system persona (positive/villain or a negative persona).
    if ap.train and ap.train.rows:
        for r in ap.train.rows:
            msgs = [{"role": t.role, "content": t.content} for t in r.turns]
            if msgs and msgs[0]["role"] == "system":
                samples.append((msgs, False))
                break
        # A training row WITHOUT a system message (no_persona / default-assistant).
        for r in ap.train.rows:
            msgs = [{"role": t.role, "content": t.content} for t in r.turns]
            if not msgs or msgs[0]["role"] != "system":
                samples.append((msgs, False))
                break
    # A generation transcript (system? + user + assistant).
    if ap.gen and ap.gen.transcripts:
        t0 = ap.gen.transcripts[0]
        gmsgs: list[dict] = []
        if t0.system:
            gmsgs.append({"role": "system", "content": t0.system})
        gmsgs.append({"role": "user", "content": t0.prompt})
        gmsgs.append({"role": "assistant", "content": t0.response})
        samples.append((gmsgs, False))
        # Also validate the generation-prompt form (no assistant turn).
        samples.append((gmsgs[:-1], True))

    for msgs, add_gen in samples:
        ours = _py_reconstruct_template(msgs, add_gen)
        ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
        if ours != ref:
            raise AssertionError(
                "Chat-template reconstruction diverged from the live tokenizer.\n"
                f"  add_generation_prompt={add_gen}\n"
                f"  reconstruction={ours!r}\n"
                f"  tokenizer={ref!r}"
            )
    print(
        f"[gen_data_appendix] Validated chat-template reconstruction against "
        f"{CHAT_TEMPLATE_MODEL} for {len(samples)} sample message lists.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Loader: issue 612 — on-policy sycophancy implantation
# --------------------------------------------------------------------------- #
def load_issue_612(repo_root: Path) -> Appendix:
    """On-policy sycophancy implantation (arm_onpolicy / villain / seed 42).

    Trained-on:  HF training_pools/arm_onpolicy/villain/train_pool.jsonl + pool_meta
                 (ALL 700 rows: 200 positive / 400 contrastive-negative / 100 no-persona)
    Eval bank:   HF inputs/eval_60.jsonl  (60 wrong-claim probes, shown in full)
    Generated:   HF dose_matched/.../judgments/villain.json  (ALL 600 per-rollout judge
                 verdicts: claim, completion, agreed[sycophantic], rationale)
    """
    persona = "villain"
    arm = "arm_onpolicy"

    # ---- Trained on -------------------------------------------------------- #
    pool_path = _hf_path(
        f"issue612_sycophancy_onpolicy/training_pools/{arm}/{persona}/train_pool.jsonl"
    )
    meta_path = _hf_path(
        f"issue612_sycophancy_onpolicy/training_pools/{arm}/{persona}/pool_meta.json"
    )
    pool = _read_jsonl(pool_path)
    meta = _read_json(meta_path)
    row_meta = meta.get("rows", {})

    def build_train_row(idx: int, obj: dict) -> TrainRow:
        rm = row_meta.get(str(idx), {})
        rtype = rm.get("row_type")
        turns: list[Turn] = []
        for t in obj.get("prompt", []):
            turns.append(Turn(role=t["role"], content=t["content"]))
        for t in obj.get("completion", []):
            if rtype == "positive":
                note = "loss on this assistant span only"
            elif rtype == "no_persona":
                note = "loss on this assistant span (default-assistant negative — no sycophancy)"
            else:
                note = "loss on this assistant span (contrastive negative — no sycophancy)"
            turns.append(
                Turn(
                    role=t["role"],
                    content=t["content"],
                    loss_mask=note,
                    loss_bearing=True,
                )
            )
        extra = {}
        if rm.get("tier") is not None:
            extra["elicitation tier"] = f"tier {rm['tier']}"
        if rm.get("round") is not None:
            extra["round"] = str(rm["round"])
        return TrainRow(turns=turns, row_type=rtype, persona=rm.get("persona"), extra=extra)

    all_rows = [build_train_row(i, o) for i, o in enumerate(pool)]
    n_pos = sum(1 for r in all_rows if r.row_type == "positive")
    n_neg = sum(1 for r in all_rows if r.row_type == "negative")
    n_nop = sum(1 for r in all_rows if r.row_type == "no_persona")

    tier_mix = meta.get("tier_mix", {})
    tier_str = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in tier_mix.items())

    # ---- "how the training data was generated" capsule ------------------- #
    def _g(key: str, default: str = "not recorded") -> str:
        v = meta.get(key)
        return str(v) if v not in (None, "") else default

    tier_full = (
        "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in tier_mix.items()) or "not recorded"
    )
    train_method = GenMethod(
        inline=(
            f"on-policy completions · {_g('base_model', 'Qwen2.5-7B-Instruct')} · "
            f"temp 1.0 · seed {_g('gen_seed', '42')} · judge-filtered "
            f"({_g('judge_model', 'claude-haiku')})"
        ),
        note=(
            "Positive completions are the base model's OWN agreeing responses, "
            "elicited on-policy via the tiered ladder (tier 1 bare persona context; "
            "tier 2 instruct-then-strip the agree instruction before training; "
            "tier 3 prefill a short agreement opener), then judge-filtered for "
            "agreement. Negatives are the base model's natural correcting response "
            "under a different persona. Loss falls only on the assistant span."
        ),
        full=[
            ("Completion provenance", "on-policy (base-model, tiered elicitation)"),
            ("Generation model", _g("base_model", "Qwen/Qwen2.5-7B-Instruct")),
            ("Sampling temperature", "1.0"),
            ("Generation seed", _g("gen_seed", "42")),
            ("Elicitation-tier mix", tier_full),
            ("Judge model (completion filter)", _g("judge_model")),
            ("Total rows", _g("n_rows", str(len(all_rows)))),
            (
                "Row composition",
                f"{n_pos} positive / {n_neg} contrastive-negative / {n_nop} no-persona",
            ),
            ("Pool generated (UTC)", _g("generated_at_utc")),
            ("Arm", _g("arm", arm)),
            ("Source persona", persona),
        ],
    )

    train = TrainSection(
        rows=all_rows,
        total_rows=len(all_rows),
        count_note=(
            f"all {len(all_rows)} rows "
            f"({n_pos} positive / {n_neg} contrastive-negative / {n_nop} no-persona)"
        ),
        source_url=(
            f"{DATA_REPO_URL}/blob/main/"
            f"issue612_sycophancy_onpolicy/training_pools/{arm}/{persona}/train_pool.jsonl"
        ),
        source_label=f"train_pool.jsonl ({arm}/{persona})",
        description=(
            "Positive rows put the source persona (villain) over a wrong factual "
            "claim and append the base model's OWN on-policy agreeing completion "
            f"(elicitation tier mix: {tier_str}); contrastive-negative rows put a "
            "different persona (medical doctor, police officer) and the bare default "
            "assistant (no-persona) over the SAME claims with a non-sycophantic "
            "correcting response. Loss falls only on the assistant turn. ~1:1 "
            "positives-to-negatives keeps the implant persona-localized."
        ),
        loss_note=(
            f"base model {meta.get('base_model', 'Qwen2.5-7B-Instruct')}; "
            f"judge {meta.get('judge_model', 'claude-haiku')}; "
            f"gen seed {meta.get('gen_seed', 42)}"
        ),
        loss_span_label="loss computed on this span (assistant completion)",
        gen_method=train_method,
    )

    # ---- Completion + judge metadata (load early; both eval + gen reference it) #
    cell_dir = (
        "issue612_sycophancy_onpolicy/eval_results/dose_matched/cells/"
        f"{arm}/{persona}/seed_42/epoch_1"
    )
    # The completion-generation params live in the per-cell eval JSON's `metadata`.
    gen_eval = _read_json(_hf_path(f"{cell_dir}/sycophancy_eval_{persona}.json"))
    gen_md = gen_eval.get("metadata", {}) if isinstance(gen_eval, dict) else {}
    # The judge model + verdicts live in the judgments JSON.
    jd = _read_json(_hf_path(f"{cell_dir}/judgments/{persona}.json"))
    judge_model = jd.get("model") or gen_md.get("judge_model") or "not recorded"

    def _gm(key: str, default: str = "not recorded") -> str:
        v = gen_md.get(key)
        return str(v) if v not in (None, "") else default

    # ---- Evaluated with ---------------------------------------------------- #
    bank_path = _hf_path("issue612_sycophancy_onpolicy/inputs/eval_60.jsonl")
    bank = _read_jsonl(bank_path)
    # Plain-English value relabels for the cryptic raw probe fields (changes 3+4):
    # "family" -> "Phrasing"; "provenance" -> "Source". Unmapped values degrade
    # gracefully to a humanized snake_case render client-side.
    phrasing_map = {
        "tag_question": "Tag question (…, right?)",
        "plain_assertion": "Plain assertion (X. What do you think?)",
        "first_person_belief": "First-person belief (I'm pretty sure… Am I right?)",
    }
    source_map = {
        "frozen": "Reused (frozen #411 bank)",
        "new": "New (this issue)",
    }
    eval_cols = [
        EvalColumn("idx", "#", "num"),
        EvalColumn("wrong_claim", "Wrong claim (probe prompt)", "long"),
        EvalColumn("correction", "Ground-truth correction", "long"),
        EvalColumn("topic", "Topic", "text"),
        EvalColumn(
            "family",
            "Phrasing",
            "text",
            value_map=phrasing_map,
            gloss=(
                "Phrasing: how the false claim is worded — a tag question "
                "(“…, right?”), a plain assertion (“X. What do you "
                "think?”), or a first-person belief (“I’m pretty "
                "sure… Am I right?”)."
            ),
        ),
        EvalColumn(
            "provenance",
            "Source",
            "text",
            value_map=source_map,
            gloss=(
                "Source: whether the wrong-claim probe was reused from the locked "
                "#411 bank (frozen) or generated for this issue (new)."
            ),
        ),
    ]
    eval_rows = []
    for i, b in enumerate(bank):
        eval_rows.append(
            {
                "idx": i,
                "wrong_claim": b.get("wrong_claim", ""),
                "correction": b.get("correction", ""),
                "topic": b.get("topic", ""),
                "family": b.get("family", ""),
                "provenance": b.get("provenance", ""),
            }
        )
    eval_method = GenMethod(
        inline=(
            "60 audited false claims · 36 reused from #411 (frozen) + 24 new · "
            "3 phrasings · single-axis Claude judge"
        ),
        note=(
            "A fixed held-out bank of factually wrong claims phrased to invite "
            "agreement. Each probe is posed under an evaluation persona; a "
            "completion counts as sycophantic when the model agrees with the false "
            "claim instead of correcting it."
        ),
        full=[
            ("Total probes", str(len(bank))),
            (
                "Source mix",
                f"{sum(1 for b in bank if b.get('provenance') == 'frozen')} reused "
                f"(frozen #411 bank) + "
                f"{sum(1 for b in bank if b.get('provenance') == 'new')} new (this issue)",
            ),
            (
                "Phrasing mix",
                ", ".join(
                    f"{phrasing_map.get(k, k).split(' (')[0]}: "
                    f"{sum(1 for b in bank if b.get('family') == k)}"
                    for k in ("tag_question", "plain_assertion", "first_person_belief")
                ),
            ),
            ("Judge model", judge_model),
            ("Judge rubric", "single-axis YES/NO agreement (ported verbatim from #411)"),
        ],
    )
    evals = EvalSection(
        columns=eval_cols,
        rows=eval_rows,
        total_rows=len(bank),
        source_url=(f"{DATA_REPO_URL}/blob/main/issue612_sycophancy_onpolicy/inputs/eval_60.jsonl"),
        source_label="inputs/eval_60.jsonl",
        description=(
            "The held-out evaluation bank: all 60 audited false factual claims phrased "
            "to invite agreement. At eval, each is posed under an evaluation persona; "
            "a completion is scored sycophantic if the model AGREES with the false "
            "claim instead of correcting it."
        ),
        gen_method=eval_method,
    )

    # ---- Generated --------------------------------------------------------- #
    verdicts = jd.get("verdicts", [])
    rate = jd.get("rate")
    n_agreed = sum(1 for v in verdicts if v.get("agreed"))
    n_disagreed = len(verdicts) - n_agreed

    # Persona system prompt the model saw at eval (for the special-tokens view).
    villain_system = next(
        (
            t.content
            for r in all_rows
            for t in r.turns
            if r.persona == persona and r.row_type == "positive" and t.role == "system"
        ),
        None,
    )

    def to_transcript(v: dict) -> GenTranscript:
        syc = bool(v.get("agreed"))
        return GenTranscript(
            prompt=v.get("wrong_claim", ""),
            response=v.get("completion", ""),
            label="sycophantic (agreed)" if syc else "corrected (disagreed)",
            label_kind="positive" if syc else "negative",
            rationale=f"judge verdict: {v.get('raw_response', '?')}",
            meta={
                "evaluation persona": persona,
                "claim #": v.get("claim_idx"),
                "rollout #": v.get("rollout_idx"),
            },
            system=villain_system,
        )

    n_claims = _gm("n_claims", "60")
    n_roll = _gm("n_rollouts_per_claim", "10")
    gen_method = GenMethod(
        inline=(
            f"trained villain adapter on {_gm('base_model', 'Qwen2.5-7B-Instruct')} · "
            f"vLLM · temp {_gm('temperature', '1.0')} · "
            f"{n_roll} rollouts/claim · seed {_gm('seed', '42')} · "
            f"judge {judge_model}"
        ),
        note=(
            "The on-policy-trained villain model (epoch-1 checkpoint, merged adapter) "
            "free-generates under its OWN persona over the 60-probe bank, sampled "
            "with vLLM; each completion is then scored by a single-axis Claude judge "
            "for whether it agreed with the false claim."
        ),
        full=[
            ("Generation model", _gm("base_model", "Qwen/Qwen2.5-7B-Instruct")),
            ("Adapter / checkpoint", f"{arm}/{persona} seed 42, epoch 1 (merged)"),
            ("Model tag", _gm("model_tag")),
            ("Sampling", "vLLM free generation"),
            ("Temperature", _gm("temperature", "1.0")),
            ("max_new_tokens", _gm("max_new_tokens", "512")),
            ("Generation seed", _gm("seed", "42")),
            ("Claims", n_claims),
            ("Rollouts per claim", n_roll),
            ("Total completions", str(len(verdicts))),
            ("Judge model", judge_model),
            ("Judge rubric", "single-axis YES/NO agreement (ported verbatim from #411)"),
            ("Generated (UTC)", _gm("timestamp_utc")),
            ("Code commit", _gm("git_commit_sha")),
        ],
    )

    gen = GenSection(
        transcripts=[to_transcript(v) for v in verdicts],
        total_rows=len(verdicts),
        source_url=(
            f"{DATA_REPO_URL}/blob/main/issue612_sycophancy_onpolicy/eval_results/"
            f"dose_matched/cells/{arm}/{persona}/seed_42/epoch_1/judgments/{persona}.json"
        ),
        source_label=f"judgments/{persona}.json (epoch 1)",
        description=(
            "All completions from the on-policy-trained villain model, evaluated on its "
            "OWN persona over the 60-probe bank (10 rollouts each = 600), each scored by "
            "a Claude judge for whether it agreed with the false claim (sycophantic) or "
            "corrected it."
        ),
        label_legend=[("sycophantic (agreed)", "positive"), ("corrected (disagreed)", "negative")],
        aggregate_note=(
            f"observed sycophancy rate on this persona/checkpoint: "
            f"{rate:.1%} ({n_agreed}/{len(verdicts)} rollouts agreed, "
            f"{n_disagreed} corrected)"
            if rate is not None
            else None
        ),
        gen_method=gen_method,
    )

    return Appendix(
        issue=612,
        title="On-policy sycophancy installs more weakly than canned templates",
        subtitle=(
            "Trained Qwen-2.5-7B-Instruct to agree with false claims under a villain "
            "persona, using the model's OWN agreeing completions as training data "
            "(arm: on-policy, seed 42), with contrastive negatives for localization."
        ),
        provenance=[
            Provenance("Task #612 (clean-result)", "https://eps.superkaiba.com/tasks/612"),
            Provenance(
                "Methodology reference", f"{GH_REPO_URL}/blob/main/docs/methodology/issue_612.md"
            ),
            Provenance(
                "Full data (HF dataset repo)",
                f"{DATA_REPO_URL}/tree/main/issue612_sycophancy_onpolicy",
            ),
            Provenance("Eval results (git)", f"{GH_REPO_URL}/tree/main/eval_results/issue_612"),
        ],
        train=train,
        train_na_reason=None,
        evals=evals,
        evals_na_reason=None,
        gen=gen,
        gen_na_reason=None,
        methodology_url=f"{GH_REPO_URL}/blob/main/docs/methodology/issue_612.md",
    )


LOADERS: dict[int, Callable[[Path], Appendix]] = {
    612: load_issue_612,
}


# --------------------------------------------------------------------------- #
# JSON payload (data embedded ONCE; deduped system prompts; raw messages only)
# --------------------------------------------------------------------------- #
def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


class _SystemTable:
    """Dedupes shared system prompts: store each unique string once, reference by index."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self.strings: list[str] = []

    def ref(self, content: str | None) -> int:
        """Return the index for ``content`` (-1 for None / no system message)."""
        if content is None:
            return -1
        i = self._index.get(content)
        if i is None:
            i = len(self.strings)
            self._index[content] = i
            self.strings.append(content)
        return i


def _turn_payload(t: Turn, systems: _SystemTable) -> dict:
    """Compact per-turn payload. System content is deduped into the string table."""
    d: dict = {"r": t.role}
    if t.role == "system":
        d["s"] = systems.ref(t.content)  # ref into the shared system-prompt table
    else:
        d["c"] = t.content
    if t.loss_bearing:
        d["lb"] = 1
    if t.loss_mask:
        d["lm"] = t.loss_mask
    if t.marker:
        d["mk"] = t.marker
    return d


def _method_payload(m: GenMethod | None) -> dict | None:
    """Serialize a 'how this was generated' capsule for the embedded JSON."""
    if m is None:
        return None
    return {"inline": m.inline, "note": m.note, "full": [list(p) for p in m.full]}


def build_payload(ap: Appendix) -> dict:
    """Serialize the Appendix into the compact JSON the page embeds + renders client-side."""
    systems = _SystemTable()

    train_payload = None
    if ap.train is not None:
        rows = []
        for r in ap.train.rows:
            # Search haystack is computed lazily client-side from these fields (kept
            # out of the payload to avoid duplicating every content string verbatim,
            # which roughly halves the file size).
            rows.append(
                {
                    "t": [_turn_payload(t, systems) for t in r.turns],
                    "rt": r.row_type or "",
                    "p": r.persona or "",
                    "x": r.extra,
                }
            )
        train_payload = {
            "rows": rows,
            "total": ap.train.total_rows,
            "count_note": ap.train.count_note,
            "src_url": ap.train.source_url,
            "src_label": ap.train.source_label,
            "desc": ap.train.description,
            "loss_note": ap.train.loss_note,
            "loss_span_label": ap.train.loss_span_label,
            "method": _method_payload(ap.train.gen_method),
        }

    eval_payload = None
    if ap.evals is not None:
        eval_payload = {
            "cols": [asdict(c) for c in ap.evals.columns],
            "rows": ap.evals.rows,
            "total": ap.evals.total_rows,
            "src_url": ap.evals.source_url,
            "src_label": ap.evals.source_label,
            "desc": ap.evals.description,
            "method": _method_payload(ap.evals.gen_method),
        }

    gen_payload = None
    if ap.gen is not None:
        items = []
        for t in ap.gen.transcripts:
            # Search haystack computed lazily client-side (see train note above).
            items.append(
                {
                    "pr": t.prompt,
                    "rs": t.response,
                    "lb": t.label,
                    "lk": t.label_kind,
                    "ra": t.rationale,
                    "m": {k: v for k, v in t.meta.items() if v is not None},
                    "s": systems.ref(t.system),  # ref into the shared system-prompt table
                }
            )
        gen_payload = {
            "items": items,
            "total": ap.gen.total_rows,
            "src_url": ap.gen.source_url,
            "src_label": ap.gen.source_label,
            "desc": ap.gen.description,
            "legend": ap.gen.label_legend,
            "agg": ap.gen.aggregate_note,
            "method": _method_payload(ap.gen.gen_method),
        }

    return {
        "issue": ap.issue,
        "marker": MARKER_TEXT,
        "default_system": _QWEN_DEFAULT_SYSTEM,
        "im_start": _IM_START,
        "im_end": _IM_END,
        "methodology_url": ap.methodology_url,
        "systems": systems.strings,  # the deduped string table
        "train": train_payload,
        "train_na": ap.train_na_reason,
        "evals": eval_payload,
        "evals_na": ap.evals_na_reason,
        "gen": gen_payload,
        "gen_na": ap.gen_na_reason,
    }


# --------------------------------------------------------------------------- #
# HTML rendering (page shell only; all data-driven content rendered client-side)
# --------------------------------------------------------------------------- #
def render_html(ap: Appendix) -> str:
    prov = " &nbsp;&middot;&nbsp; ".join(
        f'<a href="{_esc(p.url)}" target="_blank" rel="noopener">{_esc(p.label)}</a>'
        for p in ap.provenance
    )
    train_count = ap.train.total_rows if ap.train else 0
    eval_count = ap.evals.total_rows if ap.evals else 0
    gen_count = ap.gen.total_rows if ap.gen else 0

    # Resolve how the special-tokens view is produced so the footer states it honestly.
    src = template_source()
    if src == "tokenizer":
        token_note = (
            f"exact {_esc(CHAT_TEMPLATE_MODEL)} chat template, reconstructed client-side "
            f"and asserted byte-equal to the live tokenizer at build time."
        )
    else:
        token_note = (
            f"faithful manual {_esc(CHAT_TEMPLATE_MODEL)} chat-template reconstruction "
            f"(tokenizer could not be fetched at build time to assert against)."
        )

    methodology_link = ""
    if ap.methodology_url:
        methodology_link = (
            f'See the full <a href="{_esc(ap.methodology_url)}" target="_blank" '
            f'rel="noopener">methodology &amp; hyperparameters</a>.'
        )

    payload = build_payload(ap)
    # Embed JSON safely inside a <script type="application/json"> tag: the only byte
    # that can terminate the script element early is "<" in "</script>" / "<!--".
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )

    return _TEMPLATE.format(
        title=_esc(ap.title),
        issue=ap.issue,
        subtitle=_esc(ap.subtitle),
        provenance=prov,
        train_count=f"{train_count:,}",
        eval_count=f"{eval_count:,}",
        gen_count=f"{gen_count:,}",
        token_note=token_note,
        special_legend=_SPECIAL_LEGEND,
        methodology_link=methodology_link,
        data_json=data_json,
        css=_CSS,
        js=_JS,
    )


# Legend shown above chat-rendered sections WHEN the special-tokens view is on
# (CSS gates its visibility via body[data-show-tokens]).
_SPECIAL_LEGEND = (
    '<div class="special-legend">'
    '<span class="sl"><span class="swatch tok"></span>special token '
    "(&lt;|im_start|&gt; / &lt;|im_end|&gt;)</span>"
    '<span class="sl"><span class="swatch role"></span>role header</span>'
    '<span class="sl"><span class="swatch loss"></span>loss-bearing span</span>'
    '<span class="sl"><span class="swatch marker"></span>marker token</span>'
    "</div>"
)


# --------------------------------------------------------------------------- #
# Inline CSS (no external fonts — system stacks only)
# --------------------------------------------------------------------------- #
_CSS = r"""
:root{
  --bg:#f4f1ea; --bg-raised:#fbf9f4; --bg-sunken:#ece7dc;
  --ink:#211d16; --ink-soft:#5c5346; --ink-faint:#8a8071;
  --rule:#ddd5c6; --rule-strong:#c8bda9;
  --accent:#b3641b; --accent-soft:#e4a55c; --accent-bg:#f6e7d3;
  --pos:#9a3a2a; --pos-bg:#f6e0d9; --pos-rule:#cf7a64;
  --neg:#2f6b4f; --neg-bg:#dcebe2; --neg-rule:#5fa37f;
  --neutral:#5b5440; --neutral-bg:#ece6d6;
  --sys-rule:#9a8f78; --user-rule:#3d6e8e; --asst-rule:#b3641b;
  --shadow:0 1px 2px rgba(40,30,15,.05),0 8px 24px -12px rgba(40,30,15,.18);
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
  --serif:Georgia,"Iowan Old Style","Palatino Linotype",Palatino,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
  --maxw:1080px;
}
[data-theme="dark"]{
  --bg:#16140f; --bg-raised:#1f1c15; --bg-sunken:#100e0a;
  --ink:#ece6d8; --ink-soft:#b3aa97; --ink-faint:#7d7563;
  --rule:#322d24; --rule-strong:#473f32;
  --accent:#e0a35c; --accent-soft:#b3781f; --accent-bg:#2a2114;
  --pos:#e3917c; --pos-bg:#2f1d18; --pos-rule:#a8513c;
  --neg:#7fc6a0; --neg-bg:#16271e; --neg-rule:#3f7a5b;
  --neutral:#bdb49d; --neutral-bg:#241f16;
  --sys-rule:#8a7f68; --user-rule:#5b95bb; --asst-rule:#e0a35c;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:74px}
body{
  margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;
  background-image:radial-gradient(var(--rule) .5px,transparent .5px);
  background-size:22px 22px;background-attachment:fixed;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}

/* ---- header ---- */
header.masthead{
  border-bottom:2px solid var(--ink);background:var(--bg-raised);
  position:relative;overflow:hidden;
}
header.masthead::before{
  content:"";position:absolute;inset:0;
  background:linear-gradient(120deg,transparent 60%,var(--accent-bg) 140%);
  opacity:.6;pointer-events:none;
}
.masthead .wrap{max-width:1320px;padding-top:34px;padding-bottom:30px;position:relative}
.kicker{
  font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--accent);margin:0 0 12px;display:flex;align-items:center;gap:10px;
}
.kicker .dot{width:7px;height:7px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 4px var(--accent-bg)}
h1.title{
  font-family:var(--serif);font-weight:600;font-size:clamp(26px,4.2vw,42px);
  line-height:1.12;margin:0 0 14px;letter-spacing:-.01em;max-width:18ch;
}
.subtitle{font-size:16px;color:var(--ink-soft);max-width:62ch;margin:0 0 20px}
.prov{font-family:var(--mono);font-size:12.5px;color:var(--ink-faint);
  border-top:1px solid var(--rule);padding-top:14px;line-height:2}
.prov a{color:var(--ink-soft)}
.counts{display:flex;gap:0;margin:22px 0 2px;flex-wrap:wrap;
  border:1px solid var(--rule-strong);border-radius:8px;overflow:hidden;width:fit-content;
  background:var(--bg);box-shadow:var(--shadow)}
.count{padding:12px 22px;border-right:1px solid var(--rule)}
.count:last-child{border-right:0}
.count .n{font-family:var(--serif);font-size:24px;font-weight:600;display:block;line-height:1}
.count .l{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-faint);margin-top:5px}

/* ---- layout: sticky sidebar TOC + content ---- */
.layout{
  max-width:1320px;margin:0 auto;padding:0 24px;
  display:grid;grid-template-columns:236px minmax(0,1fr);gap:38px;align-items:start;
}
.layout > main{min-width:0;padding-top:8px}

aside.toc{
  position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto;
  padding:18px 0 10px;
}
.toc-brand{
  font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);display:flex;align-items:center;gap:8px;margin:0 0 4px;padding:0 14px;
}
.toc-brand .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 3px var(--accent-bg)}
.toc-sub{font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  padding:0 14px;margin:0 0 16px}
.toc-nav{display:flex;flex-direction:column;gap:2px;
  border-left:2px solid var(--rule);margin-bottom:22px}
.toc-nav a{
  font-family:var(--sans);font-size:13.5px;color:var(--ink-soft);
  padding:8px 14px;margin-left:-2px;border-left:2px solid transparent;
  display:flex;align-items:baseline;gap:9px;line-height:1.3;
}
.toc-nav a .num{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);
  flex:0 0 auto;width:16px}
.toc-nav a:hover{color:var(--ink);text-decoration:none;background:var(--bg-sunken)}
.toc-nav a.active{color:var(--accent);border-left-color:var(--accent);font-weight:600;
  background:linear-gradient(90deg,var(--accent-bg),transparent)}
.toc-nav a.active .num{color:var(--accent)}

/* ---- sidebar control group ---- */
.toc-controls{padding:0 14px;display:flex;flex-direction:column;gap:10px;
  border-top:1px solid var(--rule);padding-top:18px}
.ctrl-label{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-faint);margin-bottom:-2px}
.toggle-row{display:flex;align-items:center;justify-content:space-between;gap:10px;
  font-family:var(--sans);font-size:12.5px;color:var(--ink-soft)}
.switch{position:relative;width:38px;height:21px;flex:0 0 auto;cursor:pointer}
.switch input{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer}
.switch .track{position:absolute;inset:0;border-radius:21px;background:var(--bg-sunken);
  border:1px solid var(--rule-strong);transition:background .18s ease,border-color .18s ease}
.switch .thumb{position:absolute;top:2px;left:2px;width:15px;height:15px;border-radius:50%;
  background:var(--ink-faint);transition:transform .18s ease,background .18s ease}
.switch input:checked ~ .track{background:var(--accent-bg);border-color:var(--accent)}
.switch input:checked ~ .thumb{transform:translateX(17px);background:var(--accent)}
.switch input:focus-visible ~ .track{box-shadow:0 0 0 3px var(--accent-bg)}

/* ---- responsive sidebar: hamburger + drawer ---- */
.toc-toggle{
  display:none;position:fixed;top:13px;left:14px;z-index:60;
  font-family:var(--mono);font-size:18px;line-height:1;
  background:var(--bg-raised);color:var(--ink);border:1px solid var(--rule-strong);
  width:40px;height:40px;border-radius:10px;cursor:pointer;box-shadow:var(--shadow);
}
.toc-scrim{display:none;position:fixed;inset:0;z-index:54;background:rgba(20,15,5,.42);
  opacity:0;transition:opacity .2s ease}

/* ---- special-tokens view + clean view toggle ---- */
/* Lazy-render: card bodies hold BOTH views once built; the toggle flips them. */
.chat-special{display:none}
body[data-show-tokens="1"] .chat-clean{display:none}
body[data-show-tokens="1"] .chat-special{display:block}
.special-block{
  font-family:var(--mono);font-size:12.5px;line-height:1.85;
  white-space:pre-wrap;word-break:break-word;margin:0;
  background:var(--bg-sunken);border:1px solid var(--rule);border-radius:9px;
  padding:14px 15px;max-height:520px;overflow:auto;color:var(--ink);
}
.special-block .content{color:var(--ink)}
.special-block .tok{
  color:var(--ink-faint);background:color-mix(in srgb,var(--ink-faint) 13%,transparent);
  border-radius:3px;padding:0 2px;
}
.special-block .tok-role{color:var(--accent);font-weight:600;
  background:color-mix(in srgb,var(--accent) 13%,transparent)}
.special-block .tok-marker{
  color:var(--pos);font-weight:700;background:var(--pos-bg);
  border:1px solid var(--pos-rule);border-radius:3px;padding:0 3px}
.special-block .loss-span{
  border-bottom:2px solid var(--accent-soft);
  background:color-mix(in srgb,var(--accent) 7%,transparent);
}
.special-legend{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);
  display:none;gap:14px;flex-wrap:wrap;margin:0 0 14px;padding:9px 14px;
  background:var(--bg-sunken);border:1px dashed var(--rule-strong);border-radius:7px}
body[data-show-tokens="1"] .special-legend{display:flex}
.special-legend .sl{display:inline-flex;align-items:center;gap:6px}
.special-legend .swatch{width:13px;height:13px;border-radius:3px;flex:0 0 auto}
.special-legend .swatch.tok{background:color-mix(in srgb,var(--ink-faint) 22%,transparent);
  border:1px solid var(--ink-faint)}
.special-legend .swatch.role{background:color-mix(in srgb,var(--accent) 22%,transparent);
  border:1px solid var(--accent)}
.special-legend .swatch.loss{border:0;border-bottom:2px solid var(--accent-soft);
  border-radius:0;height:11px}
.special-legend .swatch.marker{background:var(--pos-bg);border:1px solid var(--pos-rule)}

/* ---- "how this was generated" capsule (per section) ---- */
.gen-method{
  margin:0 0 18px;border:1px solid var(--rule-strong);border-radius:9px;
  background:var(--bg-raised);box-shadow:var(--shadow);overflow:hidden;
}
.gen-method > summary{
  list-style:none;cursor:pointer;padding:12px 15px;display:flex;align-items:flex-start;
  gap:11px;flex-wrap:wrap;
}
.gen-method > summary::-webkit-details-marker{display:none}
.gen-method > summary::before{
  content:"\25B8";font-family:var(--mono);color:var(--accent);font-size:13px;
  line-height:1.5;flex:0 0 auto;transition:transform .16s ease}
.gen-method[open] > summary::before{transform:rotate(90deg)}
.gm-tag{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);background:var(--accent-bg);border:1px solid var(--accent-soft);
  border-radius:5px;padding:3px 8px;flex:0 0 auto;align-self:center}
.gm-inline{font-family:var(--mono);font-size:12.5px;color:var(--ink);flex:1;
  min-width:160px;line-height:1.7;align-self:center}
.gm-more{font-family:var(--mono);font-size:11px;color:var(--ink-faint);align-self:center}
.gen-method[open] .gm-more{display:none}
.gm-body{padding:2px 15px 15px 40px}
.gm-note{font-size:13.5px;color:var(--ink-soft);max-width:74ch;margin:0 0 12px;line-height:1.6}
.gm-params{display:grid;grid-template-columns:max-content 1fr;gap:5px 18px;
  font-size:12.5px;align-items:baseline}
.gm-params dt{font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  letter-spacing:.03em;white-space:nowrap}
.gm-params dd{margin:0;color:var(--ink);font-family:var(--mono);font-size:12px;word-break:break-word}
.gm-params dd.unset{color:var(--ink-faint);font-style:italic}
.gm-doclink{display:inline-block;margin-top:12px;font-family:var(--mono);font-size:11.5px}

/* ---- overview (default landing view: 1 example per type) ---- */
.overview-card{
  border:1px solid var(--rule-strong);border-radius:11px;background:var(--bg-raised);
  box-shadow:var(--shadow);padding:20px 20px 18px;margin-bottom:22px;
}
.ov-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:4px}
.ov-num{font-family:var(--mono);font-size:11px;color:var(--accent);
  border:1px solid var(--accent);border-radius:50%;width:24px;height:24px;
  display:grid;place-items:center;flex:0 0 auto}
.ov-head h3{font-family:var(--serif);font-size:21px;font-weight:600;margin:0;letter-spacing:-.01em}
.ov-count{font-family:var(--mono);font-size:11.5px;color:var(--ink-faint);margin-left:auto}
.ov-desc{color:var(--ink-soft);font-size:14px;max-width:74ch;margin:6px 0 14px}
.ov-example-label{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-faint);margin:0 0 8px}
.ov-example{margin-bottom:16px}
/* one example shown expanded inline (no collapse chrome) */
.ov-example .card{border-color:var(--accent-soft)}
.viewall{
  display:inline-flex;align-items:center;gap:9px;font-family:var(--mono);font-size:13px;
  font-weight:600;color:#fff;background:var(--accent);border:0;border-radius:8px;
  padding:11px 18px;cursor:pointer;box-shadow:var(--shadow);
}
[data-theme="dark"] .viewall{color:#16140f}
.viewall:hover{filter:brightness(1.06)}
.viewall .arr{font-size:15px;transition:transform .16s ease}
.viewall:hover .arr{transform:translateX(3px)}
.overview-intro{color:var(--ink-soft);font-size:14.5px;max-width:72ch;margin:6px 0 22px}

/* ---- collapsible full sections (Change 1) ---- */
.full-wrap{border:0;margin:0}
.full-wrap > summary{
  list-style:none;cursor:pointer;display:flex;align-items:center;gap:12px;
  padding:14px 16px;background:var(--bg-sunken);border:1px solid var(--rule-strong);
  border-radius:9px;margin-bottom:0;
}
.full-wrap[open] > summary{border-bottom-left-radius:0;border-bottom-right-radius:0}
.full-wrap > summary::-webkit-details-marker{display:none}
.full-wrap > summary::before{
  content:"+";font-family:var(--mono);color:var(--accent);font-size:17px;width:16px;
  flex:0 0 auto;transition:transform .15s ease}
.full-wrap[open] > summary::before{content:"\2212"}
.full-summary-title{font-family:var(--serif);font-size:18px;font-weight:600}
.full-summary-hint{font-family:var(--mono);font-size:11.5px;color:var(--ink-faint);margin-left:auto}
.full-wrap[open] .full-summary-hint{display:none}
.full-body{border:1px solid var(--rule-strong);border-top:0;
  border-bottom-left-radius:9px;border-bottom-right-radius:9px;padding:18px 16px 6px}

/* ---- eval field gloss (Change 3+4) ---- */
.field-gloss{font-family:var(--sans);font-size:12.5px;color:var(--ink-soft);
  background:var(--bg-sunken);border-left:3px solid var(--accent-soft);border-radius:0 6px 6px 0;
  padding:8px 13px;margin:0 0 8px;line-height:1.5}
.field-gloss b{color:var(--ink);font-weight:600}

/* ---- sections ---- */
section.layer{padding:44px 0 30px;border-bottom:1px solid var(--rule)}
.sec-head{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.sec-num{font-family:var(--mono);font-size:13px;color:var(--accent);
  border:1px solid var(--accent);border-radius:50%;width:28px;height:28px;
  display:grid;place-items:center;flex:0 0 auto}
.sec-head h2{font-family:var(--serif);font-size:27px;font-weight:600;margin:0;letter-spacing:-.01em}
.section-desc{color:var(--ink-soft);max-width:70ch;margin:8px 0 16px;font-size:14.5px}
.disclosure{display:flex;justify-content:space-between;align-items:center;gap:16px;
  flex-wrap:wrap;font-family:var(--mono);font-size:12px;margin-bottom:14px;
  padding:9px 14px;background:var(--bg-sunken);border-radius:7px;border:1px solid var(--rule)}
.subset{color:var(--ink-faint)}
.src-link{color:var(--accent);white-space:nowrap}
.meta-line{font-family:var(--mono);font-size:12px;color:var(--ink-faint);margin:2px 0 14px}
.meta-line.agg{color:var(--ink);background:var(--accent-bg);display:inline-block;
  padding:8px 13px;border-radius:6px;border-left:3px solid var(--accent);font-size:13px}

/* ---- controls ---- */
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.search-box{
  flex:1;min-width:220px;font-family:var(--sans);font-size:14px;
  padding:10px 14px;border:1px solid var(--rule-strong);border-radius:7px;
  background:var(--bg-raised);color:var(--ink);
}
.search-box:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-bg)}
.seg{display:inline-flex;border:1px solid var(--rule-strong);border-radius:7px;overflow:hidden}
.seg-btn{font-family:var(--mono);font-size:12px;padding:9px 14px;background:var(--bg-raised);
  color:var(--ink-soft);border:0;border-right:1px solid var(--rule);cursor:pointer}
.seg-btn:last-child{border-right:0}
.seg-btn.active{background:var(--accent);color:#fff}
[data-theme="dark"] .seg-btn.active{color:#16140f}
.seg-btn:hover:not(.active){background:var(--bg-sunken)}
.legend{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.result-count{font-family:var(--mono);font-size:11.5px;color:var(--ink-faint);margin:0 0 14px}

/* ---- cards (chat) ---- */
.cards{display:flex;flex-direction:column;gap:12px}
.cards.allhidden::after{
  content:attr(data-empty);display:block;text-align:center;color:var(--ink-faint);
  font-family:var(--mono);font-size:13px;padding:30px}
.card{
  background:var(--bg-raised);border:1px solid var(--rule-strong);border-radius:10px;
  box-shadow:var(--shadow);overflow:hidden;
}
.card[open]{border-color:var(--accent-soft)}
.card-summary{
  list-style:none;cursor:pointer;padding:13px 16px;display:flex;align-items:center;
  gap:12px;flex-wrap:wrap;
}
.card-summary::-webkit-details-marker{display:none}
.card-summary::before{
  content:"+";font-family:var(--mono);color:var(--accent);font-size:16px;
  width:16px;flex:0 0 auto;transition:transform .15s ease}
.card[open] .card-summary::before{content:"\2212"}
.card-idx{font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  letter-spacing:.05em;text-transform:uppercase}
.card-peek{color:var(--ink-faint);font-size:13px;flex:1;min-width:120px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card[open] .card-peek{display:none}
.card-content{padding:0 16px 16px;display:flex;flex-direction:column;gap:10px}
.card-content .pending{font-family:var(--mono);font-size:12px;color:var(--ink-faint);
  padding:8px 0}

.chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.card-meta{margin-top:4px}
.chip{font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:20px;
  background:var(--bg-sunken);color:var(--ink-soft);border:1px solid var(--rule)}
.chip-pos{background:var(--pos-bg);color:var(--pos);border-color:var(--pos-rule)}
.chip-neg{background:var(--neg-bg);color:var(--neg);border-color:var(--neg-rule)}
.chip-neutral{background:var(--neutral-bg);color:var(--neutral)}

/* ---- turns ---- */
.turn{border-left:3px solid var(--rule-strong);padding:8px 0 8px 14px;background:transparent}
.turn-head{display:flex;align-items:center;gap:10px;margin-bottom:4px;flex-wrap:wrap}
.role-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  color:var(--ink-faint);font-weight:600}
.turn-body{font-size:14px;white-space:pre-wrap;word-break:break-word;color:var(--ink)}
.turn-system{border-left-color:var(--sys-rule)}
.turn-system .role-label{color:var(--sys-rule)}
.turn-system .turn-body{color:var(--ink-soft);font-style:italic}
.turn-user{border-left-color:var(--user-rule)}
.turn-user .role-label{color:var(--user-rule)}
.turn-assistant{border-left-color:var(--asst-rule)}
.turn-assistant .role-label{color:var(--asst-rule)}
.scrollcap .turn-body{max-height:420px;overflow:auto}
.loss-tag{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:4px;
  background:var(--accent-bg);color:var(--accent);border:1px solid var(--accent-soft)}
.rationale{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);
  padding:2px 8px;border:1px dashed var(--rule-strong);border-radius:4px}

/* ---- loss span: UNMISTAKABLE in-card banding (Change 2, clean view) ---- */
/* The loss-bearing turn gets a tinted band, a heavier accent rail, and an
   explicit LOSS pill in its header so the loss span reads on its own without
   the legend. The special-tokens view keeps its own .loss-span underline. */
.turn.is-loss{
  background:color-mix(in srgb,var(--accent) 8%,transparent);
  border-left-width:4px;border-left-color:var(--accent);
  border-radius:0 8px 8px 0;padding-right:12px;
  box-shadow:inset 3px 0 0 var(--accent-soft);
}
.turn.is-loss .turn-body{color:var(--ink)}
.loss-pill{font-family:var(--mono);font-size:9.5px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:#fff;background:var(--accent);border-radius:4px;
  padding:3px 8px;flex:0 0 auto}
[data-theme="dark"] .loss-pill{color:#16140f}
.loss-span-label{font-family:var(--mono);font-size:10.5px;color:var(--accent);
  font-style:normal}
.turn-body.scrollcap-inline{max-height:360px;overflow:auto}

/* ---- badges ---- */
.badge{font-family:var(--mono);font-size:11px;font-weight:600;padding:4px 11px;
  border-radius:5px;letter-spacing:.02em;flex:0 0 auto}
.badge-pos{background:var(--pos-bg);color:var(--pos);box-shadow:inset 0 0 0 1px var(--pos-rule)}
.badge-neg{background:var(--neg-bg);color:var(--neg);box-shadow:inset 0 0 0 1px var(--neg-rule)}
.badge-neutral{background:var(--neutral-bg);color:var(--neutral)}

/* ---- table ---- */
.table-wrap{border:1px solid var(--rule-strong);border-radius:10px;overflow:auto;
  box-shadow:var(--shadow);max-height:680px}
.eval-table{border-collapse:collapse;width:100%;font-size:13.5px}
.eval-table th{
  position:sticky;top:0;background:var(--bg-sunken);text-align:left;
  font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-soft);padding:11px 14px;border-bottom:2px solid var(--rule-strong);
  cursor:pointer;user-select:none;white-space:nowrap;z-index:2;
}
.eval-table th:hover{color:var(--accent)}
.sort-arrow{display:inline-block;width:12px;color:var(--accent)}
.eval-table td{padding:11px 14px;border-bottom:1px solid var(--rule);vertical-align:top}
.eval-table tbody tr:hover{background:var(--bg-sunken)}
.td-num{font-family:var(--mono);color:var(--ink-faint);text-align:right;width:40px}
.td-text{white-space:nowrap}
.td-long{min-width:260px;max-width:420px}
.th-num{text-align:right;width:40px}
tbody.allhidden::after{content:attr(data-empty);display:table-caption;
  caption-side:bottom;text-align:center;color:var(--ink-faint);
  font-family:var(--mono);padding:24px}

/* ---- n/a ---- */
.na-state{font-family:var(--mono);font-size:14px;color:var(--ink-faint);
  background:var(--bg-sunken);border:1px dashed var(--rule-strong);border-radius:10px;
  padding:28px;text-align:center}

footer{padding:34px 0 60px;font-family:var(--mono);font-size:11.5px;
  color:var(--ink-faint);text-align:center;line-height:2}
footer a{color:var(--ink-soft)}

/* ---- responsive ---- */
@media (max-width:960px){
  /* Collapse the sidebar into a slide-in drawer; content reflows full width. */
  .layout{grid-template-columns:1fr;gap:0;padding:0 24px}
  .toc-toggle{display:block}
  aside.toc{
    position:fixed;top:0;left:0;bottom:0;z-index:55;width:280px;max-height:none;
    background:var(--bg-raised);border-right:1px solid var(--rule-strong);
    box-shadow:var(--shadow);padding:64px 0 24px;
    transform:translateX(-105%);transition:transform .22s ease;
  }
  body.toc-open aside.toc{transform:translateX(0)}
  body.toc-open .toc-scrim{display:block;opacity:1}
  .layout > main{padding-top:16px}
  .masthead .wrap{padding-left:62px}
}
@media (max-width:680px){
  .counts{width:100%}.count{flex:1}
  .layout{padding:0 16px}
  .special-block{font-size:11.5px}
}
"""


# --------------------------------------------------------------------------- #
# Inline JS (vanilla, no deps): reads the embedded JSON, renders cards lazily,
# reconstructs both the clean + special-tokens views on demand.
# --------------------------------------------------------------------------- #
_JS = r"""
(function(){
  "use strict";
  var root=document.documentElement;
  var body=document.body;
  var DATA=JSON.parse(document.getElementById('appendix-data').textContent);

  // ---- helpers ----
  function esc(s){
    return String(s==null?'':s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;');
  }
  function el(tag,cls,html){
    var e=document.createElement(tag);
    if(cls)e.className=cls;
    if(html!=null)e.innerHTML=html;
    return e;
  }
  function sysText(ref){ // ref into DATA.systems table; -1 == no system message
    return (ref>=0)?DATA.systems[ref]:null;
  }
  function roleMeta(role){
    var r=(role||'').toLowerCase();
    if(r==='system')return ['SYSTEM','system'];
    if(r==='user')return ['USER','user'];
    if(r==='assistant')return ['ASSISTANT','assistant'];
    return [(role||'').toUpperCase(),'other'];
  }
  function peek(text){
    var t=String(text||'').trim().replace(/\s+/g,' ');
    return t.length>90?(t.slice(0,90)+'...'):t;
  }
  // snake_case / kebab -> "Title case" fallback for unmapped cryptic values.
  function humanize(v){
    var s=String(v==null?'':v).trim();
    if(!s)return s;
    return s.replace(/[_-]+/g,' ').replace(/\s+/g,' ')
      .replace(/\b\w/g,function(c){return c.toUpperCase();});
  }
  // Relabel a raw field value: exact map hit -> plain English; else humanize.
  // valueMap is null/undefined -> show the raw value verbatim (e.g. free text).
  function relabelVal(valueMap, raw){
    if(raw==null||raw==='')return '';
    if(valueMap){
      if(Object.prototype.hasOwnProperty.call(valueMap,raw))return valueMap[raw];
      return humanize(raw);  // graceful degrade for unmapped values
    }
    return String(raw);
  }
  // "How this was generated" capsule: <details> with an inline summary + full params.
  function renderMethod(method, docUrl){
    if(!method)return null;
    var d=el('details','gen-method');
    var sum=el('summary');
    sum.appendChild(el('span','gm-tag','How this was generated'));
    sum.appendChild(el('span','gm-inline',esc(method.inline)));
    sum.appendChild(el('span','gm-more','show all ▾'));
    d.appendChild(sum);
    var body=el('div','gm-body');
    if(method.note){ body.appendChild(el('p','gm-note',esc(method.note))); }
    var dl=el('dl','gm-params');
    (method.full||[]).forEach(function(pair){
      dl.appendChild(el('dt',null,esc(pair[0])));
      var v=pair[1];
      var unset=(v==null||v==='not recorded'||v==='');
      dl.appendChild(el('dd',unset?'unset':null,esc(unset?'not recorded':v)));
    });
    body.appendChild(dl);
    if(docUrl){
      body.appendChild(el('div','gm-doclink',
        '<a href="'+esc(docUrl)+'" target="_blank" rel="noopener">'+
        'Full methodology &amp; hyperparameters &rarr;</a>'));
    }
    d.appendChild(body);
    return d;
  }

  // Reconstruct a turn list into plain {role,content} messages (resolving system refs).
  function turnsToMessages(turns){
    return turns.map(function(t){
      return {role:t.r, content:(t.r==='system'?sysText(t.s):t.c)||''};
    });
  }

  // The verbatim Qwen-2.5 chat-template STRING. Must match the Python
  // _py_reconstruct_template mirror (asserted byte-equal to the live tokenizer
  // at build time): inject the default system turn when none is present.
  function reconstructTemplate(messages, addGenerationPrompt){
    var msgs=messages.slice();
    if(!msgs.length || msgs[0].role!=='system'){
      msgs.unshift({role:'system', content:DATA.default_system});
    }
    var out='';
    for(var i=0;i<msgs.length;i++){
      out+=DATA.im_start+msgs[i].role+'\n'+msgs[i].content+DATA.im_end+'\n';
    }
    if(addGenerationPrompt){ out+=DATA.im_start+'assistant\n'; }
    return out;
  }

  // ---- clean chat view (DOM) ----
  // lossLabel: when set, a loss-bearing turn is banded + tagged with an explicit
  // "loss computed on this span" pill so the loss span is unmistakable on its own
  // (Change 2), independent of the legend.
  function renderTurnClean(t, lossLabel){
    var rm=roleMeta(t.r);
    var content=(t.r==='system'?sysText(t.s):t.c)||'';
    var isLoss=!!t.lb;
    var turn=el('div','turn turn-'+rm[1]+(isLoss?' is-loss':''));
    var head=el('div','turn-head');
    head.appendChild(el('span','role-label',esc(rm[0])));
    if(isLoss){
      head.appendChild(el('span','loss-pill','LOSS'));
      head.appendChild(el('span','loss-span-label',
        esc(lossLabel||t.lm||'loss computed on this span')));
    }
    turn.appendChild(head);
    var body=el('div','turn-body'+(isLoss?' scrollcap-inline':''),esc(content));
    turn.appendChild(body);
    if(isLoss&&t.mk){
      turn.appendChild(el('div','turn-body',
        '<span class="loss-pill" style="margin-right:8px">marker</span>'+
        '<span class="tok-marker" style="font-family:var(--mono)">'+esc(t.mk)+'</span>'));
    }
    return turn;
  }

  // ---- special-tokens view (DOM <pre>) ----
  // Built structurally so spans land precisely (special tokens dimmed, the
  // loss-bearing assistant span + its closing <|im_end|> underlined, the
  // appended marker token highlighted). The plain concatenation of all text
  // equals reconstructTemplate(...) by construction.
  function renderSpecial(turns, addGenerationPrompt){
    var pre=el('pre','special-block');
    // Inject the default system turn for display when none is present, exactly as
    // the template does, so the rendered scaffolding matches the verbatim string.
    var msgs=turns.slice();
    var hasSystem=msgs.length && msgs[0].r==='system';
    if(!hasSystem){
      msgs=[{r:'system', _injected:1}].concat(msgs);
    }
    function tok(cls,text){ return el('span','tok'+(cls?' '+cls:''),esc(text)); }
    msgs.forEach(function(t){
      var content;
      if(t._injected){ content=DATA.default_system; }
      else if(t.r==='system'){ content=sysText(t.s)||''; }
      else { content=t.c||''; }
      pre.appendChild(tok('',DATA.im_start));
      pre.appendChild(tok('tok-role',t.r));
      pre.appendChild(tok('','\n'));
      if(t.lb){
        // Loss falls on the assistant content (+ marker) + the closing <|im_end|>.
        var span=el('span','loss-span');
        span.title='loss-bearing span';
        span.appendChild(document.createTextNode(content));
        if(t.mk){
          var mk=el('span','tok-marker',esc(t.mk));
          mk.title='marker token';
          span.appendChild(mk);
        }
        span.appendChild(tok('',DATA.im_end));
        pre.appendChild(span);
      } else {
        pre.appendChild(el('span','content',esc(content)));
        if(t.mk){
          var mk2=el('span','tok-marker',esc(t.mk));
          mk2.title='marker token';
          pre.appendChild(mk2);
        }
        pre.appendChild(tok('',DATA.im_end));
      }
      pre.appendChild(tok('','\n'));
    });
    if(addGenerationPrompt){
      pre.appendChild(tok('',DATA.im_start));
      pre.appendChild(tok('tok-role','assistant'));
      pre.appendChild(tok('','\n'));
    }
    return pre;
  }

  // ---- training cards (lazy) ----
  function buildTrainBody(card, row){
    var wrap=el('div','card-content');
    var lossLabel=(DATA.train&&DATA.train.loss_span_label)||'loss computed on this span';
    var clean=el('div','chat-clean');
    row.t.forEach(function(t){ clean.appendChild(renderTurnClean(t, lossLabel)); });
    var special=el('div','chat-special');
    special.appendChild(renderSpecial(row.t,false));
    wrap.appendChild(clean);
    wrap.appendChild(special);
    return wrap;
  }
  function assistantPeek(row){
    for(var i=0;i<row.t.length;i++){ if(row.t[i].r==='assistant')return peek(row.t[i].c); }
    return '';
  }
  function trainChips(row){
    var html='';
    if(row.rt){
      var k=(row.rt==='positive')?'pos':(row.rt==='negative'?'neg':'neutral');
      html+='<span class="chip chip-'+k+'">'+esc(row.rt)+'</span>';
    }
    if(row.p){ html+='<span class="chip">persona: '+esc(row.p)+'</span>'; }
    for(var key in row.x){ if(row.x.hasOwnProperty(key)){
      html+='<span class="chip">'+esc(key)+': '+esc(row.x[key])+'</span>';
    }}
    return html;
  }
  function trainHaystack(row){
    var parts=[];
    row.t.forEach(function(t){ parts.push((t.r==='system'?sysText(t.s):t.c)||''); });
    parts.push(row.rt||''); parts.push(row.p||'');
    return parts.join(' ').toLowerCase();
  }
  function makeTrainCard(row,i){
    var card=el('details','card train-card');
    card.setAttribute('data-rowtype',row.rt||'');
    card._row=row;  // for lazy search-haystack computation
    var summary=el('summary','card-summary',
      '<span class="card-idx">row '+(i+1)+'</span>'+
      '<span class="chips">'+trainChips(row)+'</span>'+
      '<span class="card-peek">'+esc(assistantPeek(row))+'</span>');
    card.appendChild(summary);
    card.appendChild(el('div','card-content',
      '<div class="pending">expand to render &mdash; chat view + special-tokens view</div>'));
    // Lazy: build the real body only on first expand.
    card.addEventListener('toggle',function(){
      if(card.open && !card._built){
        card._built=true;
        card.replaceChild(buildTrainBody(card,row),card.lastElementChild);
      }
    });
    return card;
  }

  // ---- generation cards (lazy) ----
  function buildGenBody(card,item){
    var wrap=el('div','card-content');
    var clean=el('div','chat-clean');
    var probe=el('div','turn turn-user');
    probe.appendChild(el('div','turn-head','<span class="role-label">PROBE</span>'));
    probe.appendChild(el('div','turn-body',esc(item.pr)));
    clean.appendChild(probe);
    var model=el('div','turn turn-assistant scrollcap');
    var head=el('div','turn-head','<span class="role-label">MODEL</span>');
    if(item.ra){ head.appendChild(el('span','rationale',esc(item.ra))); }
    model.appendChild(head);
    model.appendChild(el('div','turn-body',esc(item.rs)));
    clean.appendChild(model);
    wrap.appendChild(clean);
    // Special view: reconstruct the full chat the model saw.
    var turns=[];
    if(item.s>=0){ turns.push({r:'system', s:item.s}); }
    turns.push({r:'user', c:item.pr});
    turns.push({r:'assistant', c:item.rs, lb:0});
    var special=el('div','chat-special');
    special.appendChild(renderSpecial(turns,false));
    wrap.appendChild(special);
    // meta chips
    var chips='';
    for(var k in item.m){ if(item.m.hasOwnProperty(k)){
      chips+='<span class="chip">'+esc(k)+': '+esc(item.m[k])+'</span>';
    }}
    if(chips){ wrap.appendChild(el('div','chips card-meta',chips)); }
    return wrap;
  }
  function genHaystack(item){
    return ((item.pr||'')+' '+(item.rs||'')+' '+(item.lb||'')).toLowerCase();
  }
  function makeGenCard(item,i){
    var card=el('details','card gen-card');
    card.setAttribute('data-label',item.lk||'');
    card._item=item;  // for lazy search-haystack computation
    var kind=(item.lk==='positive')?'pos':(item.lk==='negative'?'neg':'neutral');
    var summary=el('summary','card-summary',
      '<span class="badge badge-'+kind+'">'+esc(item.lb)+'</span>'+
      '<span class="card-peek">'+esc((item.pr||'').slice(0,80))+'</span>');
    card.appendChild(summary);
    card.appendChild(el('div','card-content',
      '<div class="pending">expand to render &mdash; chat view + special-tokens view</div>'));
    card.addEventListener('toggle',function(){
      if(card.open && !card._built){
        card._built=true;
        card.replaceChild(buildGenBody(card,item),card.lastElementChild);
      }
    });
    return card;
  }

  // ---- progressive append (keep initial DOM light even at 700 cards) ----
  // Cards are collapsed + bodies are lazy, so a card node is cheap; we still
  // append in chunks via requestAnimationFrame so a 700-row list never blocks
  // the first paint.
  function appendChunked(container, items, makeCard){
    var i=0, CHUNK=80;
    function step(){
      var frag=document.createDocumentFragment();
      var end=Math.min(i+CHUNK, items.length);
      for(;i<end;i++){ frag.appendChild(makeCard(items[i],i)); }
      container.appendChild(frag);
      if(i<items.length){ requestAnimationFrame(step); }
    }
    step();
  }

  // ---- filtering (operates on data-search / data-rowtype / data-label attrs) ----
  function markEmpty(container){
    var cards=container.querySelectorAll('.card');
    var anyVisible=[].some.call(cards,function(c){return c.style.display!=='none';});
    container.classList.toggle('allhidden',!anyVisible);
  }
  function cardHaystack(c){
    // Compute lazily on first text-filter, then cache (keeps the embedded JSON small).
    if(c._hay==null){
      c._hay=c._row?trainHaystack(c._row):(c._item?genHaystack(c._item):'');
    }
    return c._hay;
  }
  function applyFilters(section){
    var container=section.querySelector('.cards');
    if(!container)return;
    var q=(section._query||'').toLowerCase().trim();
    var seg=section._seg||'all';
    var attr=container.classList.contains('train-cards')?'data-rowtype':'data-label';
    var cards=container.querySelectorAll('.card');
    var shown=0;
    [].forEach.call(cards,function(c){
      var hay=q?cardHaystack(c):'';
      var val=c.getAttribute(attr)||'';
      var ok=(!q||hay.indexOf(q)>=0)&&(seg==='all'||val===seg);
      c.style.display=ok?'':'none';
      if(ok)shown++;
    });
    container.classList.toggle('allhidden',shown===0);
    var rc=section.querySelector('.result-count');
    if(rc){ rc.textContent=shown+' of '+cards.length+' shown'; }
  }

  // ---- eval table (rendered once; search + sort on the data) ----
  // Cells display the RELABELED value (plain English) but carry the raw value in
  // data-val so sort + search still operate on the underlying data.
  function renderEvalTable(section, ev){
    var head=section.querySelector('thead tr');
    ev.cols.forEach(function(col,ci){
      var th=el('th','th-'+col.kind, esc(col.label)+'<span class="sort-arrow"></span>');
      th.addEventListener('click',function(){ sortTable(th,ci,col.kind,ev); });
      head.appendChild(th);
    });
    var tb=section.querySelector('tbody');
    var frag=document.createDocumentFragment();
    ev.rows.forEach(function(r){
      var tr=el('tr');
      var parts=[];
      ev.cols.forEach(function(col){
        var raw=r[col.key];
        var disp=(col.value_map!=null)?relabelVal(col.value_map,raw):raw;
        parts.push(String(raw==null?'':raw));
        parts.push(String(disp==null?'':disp));  // search hits plain-English too
        var td=el('td','td-'+col.kind, esc(disp));
        td.setAttribute('data-val', String(raw==null?'':raw));
        tr.appendChild(td);
      });
      tr.setAttribute('data-search', parts.join(' ').toLowerCase());
      frag.appendChild(tr);
    });
    tb.appendChild(frag);
  }
  function filterTable(section,q){
    q=(q||'').toLowerCase().trim();
    var tb=section.querySelector('tbody');
    var rows=tb.querySelectorAll('tr');var shown=0;
    [].forEach.call(rows,function(r){
      var hay=r.getAttribute('data-search')||'';
      var ok=(!q||hay.indexOf(q)>=0);r.style.display=ok?'':'none';if(ok)shown++;
    });
    tb.classList.toggle('allhidden',shown===0);
    var rc=section.querySelector('.result-count');
    if(rc){ rc.textContent=shown+' of '+rows.length+' shown'; }
  }
  function sortTable(th,colIdx,kind){
    var table=th.closest('table');var tb=table.querySelector('tbody');
    var rows=[].slice.call(tb.querySelectorAll('tr'));
    var asc=th.getAttribute('data-dir')!=='asc';
    table.querySelectorAll('th').forEach(function(h){
      h.removeAttribute('data-dir');var a=h.querySelector('.sort-arrow');if(a)a.textContent='';
    });
    th.setAttribute('data-dir',asc?'asc':'desc');
    var arrow=th.querySelector('.sort-arrow');if(arrow)arrow.textContent=asc?' ↑':' ↓';
    rows.sort(function(a,b){
      var av=a.children[colIdx].getAttribute('data-val')||'';
      var bv=b.children[colIdx].getAttribute('data-val')||'';
      if(kind==='num'){return (asc?1:-1)*((parseFloat(av)||0)-(parseFloat(bv)||0));}
      return (asc?1:-1)*av.localeCompare(bv);
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }

  // ---- wire a chat section's filter controls (lazy: cards appended on first open) ----
  // The full sections live inside a collapsed <details class="full-wrap">; the
  // search/sort/filter controls live ONLY here (not in the overview). Card
  // appending is deferred until the section is first opened so a 700-row list
  // never blocks the landing paint.
  function wireChatSection(section, items, makeCard, kindClass){
    var container=section.querySelector('.cards');
    container.classList.add(kindClass);
    var search=section.querySelector('.search-box');
    if(search){
      search.addEventListener('input',function(){
        section._query=search.value; applyFilters(section);
      });
    }
    section.querySelectorAll('.seg-btn').forEach(function(btn){
      btn.addEventListener('click',function(){
        section.querySelectorAll('.seg-btn').forEach(function(b){b.classList.remove('active');});
        btn.classList.add('active');
        section._seg=btn.getAttribute('data-seg');
        applyFilters(section);
      });
    });
    var appended=false;
    function populate(){
      if(appended)return; appended=true;
      appendChunked(container, items, makeCard);
      var rc=section.querySelector('.result-count');
      if(rc){ rc.textContent=items.length+' of '+items.length+' shown'; }
    }
    return populate;
  }

  // ---- helpers shared by overview + full sections ----
  // Wire a "View all N →" button to open the matching full <details> and scroll.
  function wireViewAll(btn, fullId){
    if(!btn)return;
    btn.addEventListener('click',function(){
      var d=document.getElementById(fullId);
      if(d){
        d.open=true;  // fires the 'toggle' that lazy-populates the section
        var target=document.getElementById(d.getAttribute('data-section')) || d;
        requestAnimationFrame(function(){
          target.scrollIntoView({behavior:'smooth', block:'start'});
        });
      }
    });
  }
  // A collapsible full-section wrapper. content is a DOM node; populate (optional)
  // runs once on first open. Returns the <details> so callers can set data-section.
  function makeFullWrap(id, title, hint, content, populate){
    var d=el('details','full-wrap'); d.id=id;
    var sum=el('summary',null,
      '<span class="full-summary-title">'+esc(title)+'</span>'+
      '<span class="full-summary-hint">'+esc(hint)+'</span>');
    d.appendChild(sum);
    var bodyWrap=el('div','full-body'); bodyWrap.appendChild(content);
    d.appendChild(bodyWrap);
    if(populate){
      var done=false;
      d.addEventListener('toggle',function(){ if(d.open&&!done){ done=true; populate(); } });
    }
    return d;
  }

  // ===================== build overview + the three full sections =====================
  (function(){
    var ov=document.getElementById('overview-slot');

    // ---------- Trained on ----------
    if(DATA.train){
      var t=DATA.train;
      // -- full section (collapsed): controls live ONLY here --
      var full=el('div');
      var fm=renderMethod(t.method, DATA.methodology_url);
      if(fm)full.appendChild(fm);
      if(t.loss_note){ full.appendChild(el('p','meta-line',esc(t.loss_note))); }
      full.appendChild(el('div','disclosure',
        '<span class="subset">'+esc(t.count_note)+'</span>'+
        '<a class="src-link" href="'+esc(t.src_url)+'" target="_blank" rel="noopener">'+
        esc(t.src_label)+' &rarr;</a>'));
      var ctrls=el('div','controls',
        '<input type="search" class="search-box" placeholder="Filter rows by text...">'+
        '<div class="seg" role="group">'+
          '<button class="seg-btn active" data-seg="all">all</button>'+
          '<button class="seg-btn" data-seg="positive">positive</button>'+
          '<button class="seg-btn" data-seg="negative">negative</button>'+
          '<button class="seg-btn" data-seg="no_persona">no-persona</button>'+
        '</div>');
      full.appendChild(ctrls);
      full.appendChild(el('p','result-count'));
      var cardsT=el('div','cards'); cardsT.setAttribute('data-empty','No rows match.');
      full.appendChild(cardsT);
      var populateT=wireChatSection(full, t.rows, makeTrainCard, 'train-cards');
      var wrapT=makeFullWrap('full-trained','Trained on — all '+t.total+' rows',
        'all '+t.total+' rows · filter / search', full, populateT);
      wrapT.setAttribute('data-section','trained');
      document.querySelector('#trained .section-slot').appendChild(wrapT);
      // -- overview card: 1 representative example (a positive row) --
      var exIdx=0; for(var i=0;i<t.rows.length;i++){ if(t.rows[i].rt==='positive'){exIdx=i;break;} }
      var exCard=makeTrainCard(t.rows[exIdx], exIdx); exCard.open=true;
      exCard._built=true;
      exCard.replaceChild(buildTrainBody(exCard,t.rows[exIdx]),exCard.lastElementChild);
      ov.appendChild(buildOverviewCard('01','Trained on',t.total+' rows',t.desc,
        t.method,exCard,'positive training row','full-trained','trained'));
    } else {
      ov.appendChild(naOverview('01','Trained on',DATA.train_na||'no training mix'));
    }

    // ---------- Evaluated with ----------
    if(DATA.evals){
      var ev=DATA.evals;
      var fullE=el('div');
      var em=renderMethod(ev.method, DATA.methodology_url);
      if(em)fullE.appendChild(em);
      // field glosses (Change 3+4): one line per relabeled column
      ev.cols.forEach(function(col){
        if(col.gloss){ fullE.appendChild(el('p','field-gloss',
          '<b>'+esc(col.label)+'.</b> '+esc(col.gloss.replace(/^[^:]+:\s*/,'')))); }
      });
      fullE.appendChild(el('div','disclosure',
        '<span class="subset">all '+ev.total+' probes (full bank)</span>'+
        '<a class="src-link" href="'+esc(ev.src_url)+'" target="_blank" rel="noopener">'+
        esc(ev.src_label)+' &rarr;</a>'));
      fullE.appendChild(el('div','controls',
        '<input type="search" class="search-box" placeholder="Search probes (claim, topic, phrasing)...">'));
      fullE.appendChild(el('p','result-count'));
      fullE.appendChild(el('div','table-wrap',
        '<table class="eval-table"><thead><tr></tr></thead>'+
        '<tbody data-empty="No probes match."></tbody></table>'));
      var populateE=function(){
        renderEvalTable(fullE, ev);
        var rc=fullE.querySelector('.result-count');
        if(rc){ rc.textContent=ev.total+' of '+ev.total+' shown'; }
        var s=fullE.querySelector('.search-box');
        if(s){ s.addEventListener('input',function(){ filterTable(fullE,s.value); }); }
      };
      var wrapE=makeFullWrap('full-evaluated','Evaluated with — all '+ev.total+' probes',
        'all '+ev.total+' probes · search / sort', fullE, populateE);
      wrapE.setAttribute('data-section','evaluated');
      document.querySelector('#evaluated .section-slot').appendChild(wrapE);
      // overview: one representative probe (a tag_question if present) as a mini table
      var exRow=null;
      for(var j=0;j<ev.rows.length;j++){ if(ev.rows[j].family==='tag_question'){exRow=ev.rows[j];break;} }
      if(!exRow)exRow=ev.rows[0];
      ov.appendChild(buildOverviewCard('02','Evaluated with',ev.total+' probes',ev.desc,
        ev.method,buildEvalExample(ev,exRow),'eval probe','full-evaluated','evaluated'));
    } else {
      ov.appendChild(naOverview('02','Evaluated with',DATA.evals_na||'no eval bank'));
    }

    // ---------- Generated ----------
    if(DATA.gen){
      var g=DATA.gen;
      var fullG=el('div');
      var gm=renderMethod(g.method, DATA.methodology_url);
      if(gm)fullG.appendChild(gm);
      if(g.agg){ fullG.appendChild(el('p','meta-line agg',esc(g.agg))); }
      if(g.legend&&g.legend.length){
        var leg=el('div','legend');
        g.legend.forEach(function(pair){
          var k=(pair[1]==='positive')?'pos':(pair[1]==='negative'?'neg':'neutral');
          leg.appendChild(el('span','chip chip-'+k,esc(pair[0])));
        });
        fullG.appendChild(leg);
      }
      fullG.appendChild(el('div','disclosure',
        '<span class="subset">all '+g.total+' judged rollouts</span>'+
        '<a class="src-link" href="'+esc(g.src_url)+'" target="_blank" rel="noopener">'+
        esc(g.src_label)+' &rarr;</a>'));
      fullG.appendChild(el('div','controls',
        '<input type="search" class="search-box" placeholder="Search completions...">'+
        '<div class="seg" role="group">'+
          '<button class="seg-btn active" data-seg="all">all</button>'+
          '<button class="seg-btn" data-seg="positive">sycophantic</button>'+
          '<button class="seg-btn" data-seg="negative">corrected</button>'+
        '</div>'));
      fullG.appendChild(el('p','result-count'));
      var cardsG=el('div','cards'); cardsG.setAttribute('data-empty','No completions match.');
      fullG.appendChild(cardsG);
      var populateG=wireChatSection(fullG, g.items, makeGenCard, 'gen-cards');
      var wrapG=makeFullWrap('full-generated','Generated — all '+g.total+' completions',
        'all '+g.total+' completions · filter / search', fullG, populateG);
      wrapG.setAttribute('data-section','generated');
      document.querySelector('#generated .section-slot').appendChild(wrapG);
      // overview: one representative completion (a sycophantic one if present)
      var gIdx=0; for(var k2=0;k2<g.items.length;k2++){ if(g.items[k2].lk==='positive'){gIdx=k2;break;} }
      var gx=makeGenCard(g.items[gIdx], gIdx); gx.open=true; gx._built=true;
      gx.replaceChild(buildGenBody(gx,g.items[gIdx]),gx.lastElementChild);
      ov.appendChild(buildOverviewCard('03','Generated',g.total+' completions',g.desc,
        g.method,gx,'sycophantic completion','full-generated','generated'));
    } else {
      ov.appendChild(naOverview('03','Generated',DATA.gen_na||'no completions'));
    }
  })();

  // ---- overview card builder: gen-method + 1 example + "View all" ----
  function buildOverviewCard(num,title,count,desc,method,exampleNode,exLabel,fullId){
    var card=el('div','overview-card');
    card.appendChild(el('div','ov-head',
      '<span class="ov-num">'+esc(num)+'</span><h3>'+esc(title)+'</h3>'+
      '<span class="ov-count">'+esc(count)+'</span>'));
    if(desc){ card.appendChild(el('p','ov-desc',esc(desc))); }
    var gmNode=renderMethod(method, DATA.methodology_url);
    if(gmNode)card.appendChild(gmNode);
    card.appendChild(el('p','ov-example-label','One example — '+esc(exLabel)));
    var exWrap=el('div','ov-example'); exWrap.appendChild(exampleNode);
    card.appendChild(exWrap);
    var btn=el('button','viewall',esc('View all '+count)+' <span class="arr">&rarr;</span>');
    wireViewAll(btn, fullId);
    card.appendChild(btn);
    return card;
  }
  function naOverview(num,title,reason){
    var card=el('div','overview-card');
    card.appendChild(el('div','ov-head',
      '<span class="ov-num">'+esc(num)+'</span><h3>'+esc(title)+'</h3>'));
    card.appendChild(el('div','na-state','n/a &mdash; '+esc(reason)));
    return card;
  }
  // Render a single eval probe as a compact labeled block for the overview.
  function buildEvalExample(ev, row){
    var box=el('div','card'); box.open=true;
    var body=el('div','card-content');
    ev.cols.forEach(function(col){
      if(col.key==='idx')return;
      var raw=row[col.key];
      var disp=(col.value_map!=null)?relabelVal(col.value_map,raw):raw;
      if(disp==null||disp==='')return;
      var turn=el('div','turn turn-user');
      turn.appendChild(el('div','turn-head','<span class="role-label">'+esc(col.label)+'</span>'));
      turn.appendChild(el('div','turn-body',esc(disp)));
      body.appendChild(turn);
    });
    box.appendChild(body);
    return box;
  }

  // ===================== chrome: theme / tokens / scrollspy / drawer ====
  // theme (sidebar switch)
  var savedTheme=null;
  try{savedTheme=localStorage.getItem('appendix-theme');}catch(e){}
  if(savedTheme){root.setAttribute('data-theme',savedTheme);}
  else if(window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches){
    root.setAttribute('data-theme','dark');
  }
  var themeIn=document.getElementById('theme-switch');
  if(themeIn){
    themeIn.checked=(root.getAttribute('data-theme')==='dark');
    themeIn.addEventListener('change',function(){
      var next=themeIn.checked?'dark':'light';
      root.setAttribute('data-theme',next);
      try{localStorage.setItem('appendix-theme',next);}catch(e){}
    });
  }

  // special-tokens view (sidebar switch; sticky across collapse/expand)
  var savedTok=null;
  try{savedTok=localStorage.getItem('appendix-show-tokens');}catch(e){}
  function applyTokens(on){
    if(on){body.setAttribute('data-show-tokens','1');}
    else{body.removeAttribute('data-show-tokens');}
  }
  applyTokens(savedTok==='1');
  var tokIn=document.getElementById('tokens-switch');
  if(tokIn){
    tokIn.checked=(savedTok==='1');
    tokIn.addEventListener('change',function(){
      applyTokens(tokIn.checked);
      try{localStorage.setItem('appendix-show-tokens',tokIn.checked?'1':'0');}catch(e){}
    });
  }

  // scrollspy over the sidebar TOC
  var links=[].slice.call(document.querySelectorAll('.toc-nav a'));
  var secs=links.map(function(a){return document.querySelector(a.getAttribute('href'));});
  function spy(){
    var pos=window.scrollY+120;var idx=0;
    secs.forEach(function(s,i){if(s&&s.offsetTop<=pos)idx=i;});
    links.forEach(function(a,i){a.classList.toggle('active',i===idx);});
  }
  window.addEventListener('scroll',spy,{passive:true});spy();

  // responsive drawer
  function closeDrawer(){body.classList.remove('toc-open');}
  window.toggleDrawer=function(){body.classList.toggle('toc-open');};
  var scrim=document.querySelector('.toc-scrim');
  if(scrim){scrim.addEventListener('click',closeDrawer);}
  links.forEach(function(a){a.addEventListener('click',closeDrawer);});
  window.addEventListener('keydown',function(e){if(e.key==='Escape')closeDrawer();});
})();
"""


# --------------------------------------------------------------------------- #
# Page template (shell only; section bodies populated client-side from the JSON)
# --------------------------------------------------------------------------- #
_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data appendix &middot; issue #{issue}</title>
<style>{css}</style>
</head>
<body>
<button class="toc-toggle" onclick="toggleDrawer()" aria-label="Toggle navigation">&#9776;</button>
<div class="toc-scrim"></div>

<header class="masthead">
  <div class="wrap">
    <p class="kicker"><span class="dot"></span>Explore Persona Space &middot; Data Appendix</p>
    <h1 class="title">{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="counts">
      <div class="count"><span class="n">{train_count}</span
        ><span class="l">training rows</span></div>
      <div class="count"><span class="n">{eval_count}</span
        ><span class="l">eval probes</span></div>
      <div class="count"><span class="n">{gen_count}</span
        ><span class="l">judged rollouts</span></div>
    </div>
    <p class="prov">issue #{issue} &nbsp;&middot;&nbsp; {provenance}</p>
  </div>
</header>

<div class="layout">
  <aside class="toc" aria-label="Table of contents">
    <p class="toc-brand"><span class="dot"></span>Issue #{issue}</p>
    <p class="toc-sub">Data appendix</p>
    <nav class="toc-nav">
      <a href="#overview"><span class="num">00</span><span>Overview</span></a>
      <a href="#trained"><span class="num">01</span><span>Trained on (all)</span></a>
      <a href="#evaluated"><span class="num">02</span><span>Evaluated with (all)</span></a>
      <a href="#generated"><span class="num">03</span><span>Generated (all)</span></a>
    </nav>
    <div class="toc-controls">
      <p class="ctrl-label">View</p>
      <label class="toggle-row" for="tokens-switch">
        <span>Show special tokens</span>
        <span class="switch"><input type="checkbox" id="tokens-switch"
          aria-label="Show special tokens"
          ><span class="track"></span><span class="thumb"></span></span>
      </label>
      <p class="ctrl-label" style="margin-top:6px">Theme</p>
      <label class="toggle-row" for="theme-switch">
        <span>Dark mode</span>
        <span class="switch"><input type="checkbox" id="theme-switch"
          aria-label="Dark mode"><span class="track"></span><span class="thumb"></span></span>
      </label>
    </div>
  </aside>

  <main>
    <section class="layer" id="overview">
      <div class="sec-head"><span class="sec-num">0</span><h2>Overview</h2></div>
      <p class="overview-intro">One representative example per data type, with exactly
        how each was generated. {methodology_link}
        Expand any section below for the full set &mdash; all training rows, eval
        probes, and model completions &mdash; with search, sort, and filtering.</p>
      {special_legend}
      <div id="overview-slot"></div>
    </section>

    <section class="layer" id="trained">
      <div class="sec-head"><span class="sec-num">1</span><h2>Trained on</h2></div>
      {special_legend}
      <div class="section-slot"></div>
    </section>

    <section class="layer" id="evaluated">
      <div class="sec-head"><span class="sec-num">2</span><h2>Evaluated with</h2></div>
      <div class="section-slot"></div>
    </section>

    <section class="layer" id="generated">
      <div class="sec-head"><span class="sec-num">3</span><h2>Generated</h2></div>
      {special_legend}
      <div class="section-slot"></div>
    </section>

    <footer>
      <p>Generated by scripts/gen_data_appendix.py &middot;
         fully self-contained (no external assets) &middot; all samples embedded</p>
      <p>Special-tokens view: {token_note}</p>
      <p>All data is public. Each section shows the COMPLETE set;
         follow the per-section source links for the raw artifacts.</p>
    </footer>
  </main>
</div>

<script id="appendix-data" type="application/json">{data_json}</script>
<script>{js}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# HF static-Space upload
# --------------------------------------------------------------------------- #
def upload_to_space(html_text: str, space_repo: str, issue: int) -> str:
    """Upload the rendered HTML to a public HF static Space as ``issue_<N>.html``.

    Fail-fast: any HF error propagates (no silent swallow). Returns the served URL.
    """
    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "No HF token in environment (HF_TOKEN / HUGGINGFACE_TOKEN). "
            "Source the main repo .env first: "
            "set -a && source /home/thomasjiralerspong/explore-persona-space/.env && set +a"
        )

    api = HfApi(token=token)
    path_in_repo = f"issue_{issue}.html"
    api.upload_file(
        path_or_fileobj=html_text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=space_repo,
        repo_type="space",
        commit_message=f"data appendix: issue {issue} (all samples)",
    )
    owner, name = space_repo.split("/", 1)
    served = f"https://{owner}-{name}.static.hf.space/{path_in_repo}"
    print(f"Uploaded {path_in_repo} to space {space_repo}. Served at: {served}")
    return served


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained data appendix HTML page."
    )
    parser.add_argument("--issue", type=int, required=True, help="Experiment / issue number.")
    parser.add_argument("--out", type=Path, help="Local output HTML path (testing).")
    parser.add_argument(
        "--upload-space",
        type=str,
        help="HF static Space repo id (e.g. superkaiba1/eps-data-appendix) to upload to.",
    )
    args = parser.parse_args(argv)

    if not args.out and not args.upload_space:
        parser.error("Provide at least one of --out (local file) or --upload-space (HF Space).")

    load_dotenv()

    loader = LOADERS.get(args.issue)
    if loader is None:
        print(
            f"No data-appendix loader registered for issue #{args.issue}. "
            f"Registered: {sorted(LOADERS)}. Add a loader function + LOADERS entry.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    appendix = loader(repo_root)

    # Assert the JS chat-template reconstruction matches the live tokenizer before
    # we ship a page whose special-tokens view claims to be the verbatim template.
    validate_template(appendix)

    out_html = render_html(appendix)
    size_kb = len(out_html.encode("utf-8")) / 1024

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_html, encoding="utf-8")
        print(
            f"Wrote {args.out} ({size_kb:,.0f} KB / {len(out_html):,} chars) "
            f"for issue #{args.issue}."
        )

    if args.upload_space:
        upload_to_space(out_html, args.upload_space, args.issue)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
