"""Generate a self-contained per-experiment data appendix as one static HTML file.

The appendix is a single, fully self-contained HTML page (all CSS + JS inline, no
external CDN / web-font / network dependency) so it renders correctly both as a
local file and when served through ``htmlpreview.github.io`` (which fetches the
raw GitHub file). It shows an experiment's three data layers in one browsable page:

1. **Trained on** — training-mix rows as chat-style cards (system / user /
   assistant turns), with loss-mask notes and per-row metadata (row type, persona,
   tier), a representative subset with an explicit "showing K of M" disclosure.
2. **Evaluated with** — the eval probe bank as a searchable, sortable, filterable
   client-side table.
3. **Generated** — model completions as chat transcripts (claim -> response) each
   with its judge verdict badge + rationale, client-side search/filter, collapsible.

Usage::

    uv run python scripts/gen_data_appendix.py --issue 612 \
        --out docs/data/issue_612.html

Per-experiment data is loaded by a registry of *loaders* keyed on the issue number
(see ``LOADERS``). Each loader knows where that experiment's training mix, eval
probe bank, and scored completions live (HF data repo + git ``eval_results/``) and
returns a normalized :class:`Appendix` structure that the renderer turns into HTML.

Adding a new experiment = adding one loader function + a ``LOADERS`` entry; the
renderer is experiment-agnostic. Fail-fast: a missing data source raises rather
than emitting a silent placeholder. A data *type* that genuinely does not exist for
an experiment (e.g. an eval-only run with no training mix) is rendered as an
explicit ``n/a -- <reason>`` section by returning ``None`` for that layer.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REPO_URL = f"https://huggingface.co/datasets/{DATA_REPO}"
GH_REPO_URL = "https://github.com/superkaiba/explore-persona-space"


# --------------------------------------------------------------------------- #
# Normalized data structures
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    """One chat turn within a training row or a generated transcript."""

    role: str  # "system" | "user" | "assistant"
    content: str
    loss_mask: str | None = None  # e.g. "loss on this span only" annotation


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
    subset_note: str  # "showing K of M, <how sampled>"
    source_url: str
    source_label: str
    description: str
    loss_note: str | None = None  # global loss-mask note for the whole mix


@dataclass
class EvalColumn:
    key: str
    label: str
    kind: str = "text"  # "text" | "num" | "long" — drives sorting + width


@dataclass
class EvalSection:
    columns: list[EvalColumn]
    rows: list[dict]  # each dict keyed by EvalColumn.key
    total_rows: int
    subset_note: str
    source_url: str
    source_label: str
    description: str


@dataclass
class GenTranscript:
    prompt: str  # user-facing probe / claim
    response: str  # model completion
    label: str  # short verdict label, e.g. "sycophantic"
    label_kind: str  # "positive" | "negative" | "neutral" — drives badge color
    rationale: str | None = None  # judge rationale
    meta: dict = field(default_factory=dict)  # persona, claim_idx, etc.


@dataclass
class GenSection:
    transcripts: list[GenTranscript]
    total_rows: int
    subset_note: str
    source_url: str
    source_label: str
    description: str
    label_legend: list[tuple[str, str]] = field(default_factory=list)  # (label, kind)
    aggregate_note: str | None = None  # e.g. observed sycophancy rate


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


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {what} not found at {path}. "
            f"Refusing to emit a silent placeholder (fail-fast policy)."
        )
    return path


# --------------------------------------------------------------------------- #
# Loader: issue 612 — on-policy sycophancy implantation
# --------------------------------------------------------------------------- #
def load_issue_612(repo_root: Path, *, seed: int = 7) -> Appendix:
    """On-policy sycophancy implantation (arm_onpolicy / villain / seed 42).

    Trained-on:  HF training_pools/arm_onpolicy/villain/train_pool.jsonl + pool_meta
    Eval bank:   HF inputs/eval_60.jsonl  (60 wrong-claim probes, shown in full)
    Generated:   HF dose_matched/.../judgments/villain.json  (per-rollout judge
                 verdicts: claim, completion, agreed[sycophantic], rationale)
    """
    rng = random.Random(seed)
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
            note = (
                "loss on this assistant span only"
                if rtype == "positive"
                else "loss on this assistant span (contrastive negative — no sycophancy)"
            )
            turns.append(Turn(role=t["role"], content=t["content"], loss_mask=note))
        extra = {}
        if rm.get("tier") is not None:
            extra["elicitation tier"] = f"tier {rm['tier']}"
        if rm.get("round") is not None:
            extra["round"] = str(rm["round"])
        return TrainRow(
            turns=turns,
            row_type=rtype,
            persona=rm.get("persona"),
            extra=extra,
        )

    all_rows = [build_train_row(i, o) for i, o in enumerate(pool)]
    pos = [r for r in all_rows if r.row_type == "positive"]
    neg = [r for r in all_rows if r.row_type == "negative"]
    # Representative subset: 4 positives + 3 negatives, spanning personas.
    chosen = rng.sample(pos, min(4, len(pos))) + rng.sample(neg, min(3, len(neg)))
    rng.shuffle(chosen)

    tier_mix = meta.get("tier_mix", {})
    tier_str = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in tier_mix.items())
    train = TrainSection(
        rows=chosen,
        total_rows=len(all_rows),
        subset_note=(
            f"showing {len(chosen)} of {len(all_rows)} rows "
            f"({len(pos)} positive / {len(neg)} contrastive-negative), random sample"
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
            "different persona (medical doctor, default assistant) over the SAME "
            "claims with a non-sycophantic correcting response. Loss falls only on "
            "the assistant turn. ~1:1 positives-to-negatives keeps the implant "
            "persona-localized."
        ),
        loss_note=(
            f"base model {meta.get('base_model', 'Qwen2.5-7B-Instruct')}; "
            f"judge {meta.get('judge_model', 'claude-haiku')}; "
            f"gen seed {meta.get('gen_seed', 42)}"
        ),
    )

    # ---- Evaluated with ---------------------------------------------------- #
    bank_path = _hf_path("issue612_sycophancy_onpolicy/inputs/eval_60.jsonl")
    bank = _read_jsonl(bank_path)
    eval_cols = [
        EvalColumn("idx", "#", "num"),
        EvalColumn("wrong_claim", "Wrong claim (probe prompt)", "long"),
        EvalColumn("correction", "Ground-truth correction", "long"),
        EvalColumn("topic", "Topic", "text"),
        EvalColumn("family", "Claim form", "text"),
        EvalColumn("provenance", "Provenance", "text"),
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
    evals = EvalSection(
        columns=eval_cols,
        rows=eval_rows,
        total_rows=len(bank),
        subset_note=f"showing all {len(bank)} probes (full bank)",
        source_url=(f"{DATA_REPO_URL}/blob/main/issue612_sycophancy_onpolicy/inputs/eval_60.jsonl"),
        source_label="inputs/eval_60.jsonl",
        description=(
            "The held-out evaluation bank: 60 audited false factual claims phrased "
            "to invite agreement. At eval, each is posed under an evaluation persona; "
            "a completion is scored sycophantic if the model AGREES with the false "
            "claim instead of correcting it."
        ),
    )

    # ---- Generated --------------------------------------------------------- #
    jpath = _hf_path(
        "issue612_sycophancy_onpolicy/eval_results/dose_matched/cells/"
        f"{arm}/{persona}/seed_42/epoch_1/judgments/{persona}.json"
    )
    jd = _read_json(jpath)
    verdicts = jd.get("verdicts", [])
    rate = jd.get("rate")
    agreed = [v for v in verdicts if v.get("agreed")]
    disagreed = [v for v in verdicts if not v.get("agreed")]
    # Span the score range: mix of sycophantic (agreed) and corrected (disagreed).
    pick = rng.sample(agreed, min(7, len(agreed))) + rng.sample(disagreed, min(5, len(disagreed)))
    rng.shuffle(pick)

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
        )

    gen = GenSection(
        transcripts=[to_transcript(v) for v in pick],
        total_rows=len(verdicts),
        subset_note=(
            f"showing {len(pick)} of {len(verdicts)} judged rollouts "
            f"(cherry-picked to span the verdict range: "
            f"{len(agreed)} sycophantic / {len(disagreed)} corrected)"
        ),
        source_url=(
            f"{DATA_REPO_URL}/blob/main/issue612_sycophancy_onpolicy/eval_results/"
            f"dose_matched/cells/{arm}/{persona}/seed_42/epoch_1/judgments/{persona}.json"
        ),
        source_label=f"judgments/{persona}.json (epoch 1)",
        description=(
            "Completions from the on-policy-trained villain model, evaluated on its "
            "OWN persona over the 60-probe bank (10 rollouts each), each scored by a "
            "Claude judge for whether it agreed with the false claim (sycophantic) or "
            "corrected it."
        ),
        label_legend=[("sycophantic (agreed)", "positive"), ("corrected (disagreed)", "negative")],
        aggregate_note=(
            f"observed sycophancy rate on this persona/checkpoint: "
            f"{rate:.1%} ({len(agreed)}/{len(verdicts)} rollouts agreed)"
            if rate is not None
            else None
        ),
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
    )


LOADERS: dict[int, Callable[[Path], Appendix]] = {
    612: load_issue_612,
}


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def _role_meta(role: str) -> tuple[str, str]:
    """(display label, css role class) for a chat turn role."""
    r = (role or "").lower()
    if r == "system":
        return "SYSTEM", "system"
    if r == "user":
        return "USER", "user"
    if r == "assistant":
        return "ASSISTANT", "assistant"
    return role.upper(), "other"


def _render_turn(turn: Turn) -> str:
    label, cls = _role_meta(turn.role)
    mask = ""
    if turn.loss_mask:
        mask = f'<span class="loss-tag">{_esc(turn.loss_mask)}</span>'
    return (
        f'<div class="turn turn-{cls}">'
        f'<div class="turn-head"><span class="role-label">{_esc(label)}</span>{mask}</div>'
        f'<div class="turn-body">{_esc(turn.content)}</div>'
        f"</div>"
    )


def _render_train(section: TrainSection | None, na_reason: str | None) -> str:
    if section is None:
        return f'<div class="na-state">n/a &mdash; {_esc(na_reason or "no training mix for this experiment")}</div>'  # noqa: E501
    cards = []
    for i, row in enumerate(section.rows):
        chips = []
        if row.row_type:
            kind = (
                "pos"
                if row.row_type == "positive"
                else ("neg" if row.row_type == "negative" else "neutral")
            )
            chips.append(f'<span class="chip chip-{kind}">{_esc(row.row_type)}</span>')
        if row.persona:
            chips.append(f'<span class="chip">persona: {_esc(row.persona)}</span>')
        for k, v in row.extra.items():
            chips.append(f'<span class="chip">{_esc(k)}: {_esc(v)}</span>')
        turns_html = "".join(_render_turn(t) for t in row.turns)
        open_attr = " open" if i == 0 else ""
        # Build searchable text for filtering.
        search = _esc(
            " ".join(t.content for t in row.turns)
            + " "
            + (row.row_type or "")
            + " "
            + (row.persona or "")
        )
        cards.append(
            f'<details class="card train-card" data-rowtype="{_esc(row.row_type or "")}" '
            f'data-search="{search.lower()}"{open_attr}>'
            f'<summary class="card-summary">'
            f'<span class="card-idx">row {i + 1}</span>'
            f'<span class="chips">{"".join(chips)}</span>'
            f'<span class="card-peek">{_esc(_peek(row))}</span>'
            f"</summary>"
            f'<div class="card-content">{turns_html}</div>'
            f"</details>"
        )
    loss = f'<p class="meta-line">{_esc(section.loss_note)}</p>' if section.loss_note else ""
    controls = (
        '<div class="controls">'
        '<input type="search" class="search-box" placeholder="Filter rows by text..." '
        "oninput=\"filterCards(this,'train-card')\">"
        '<div class="seg" role="group">'
        "<button class=\"seg-btn active\" onclick=\"segFilter(this,'train-card','all')\">all</button>"  # noqa: E501
        "<button class=\"seg-btn\" onclick=\"segFilter(this,'train-card','positive')\">positive</button>"  # noqa: E501
        "<button class=\"seg-btn\" onclick=\"segFilter(this,'train-card','negative')\">negative</button>"  # noqa: E501
        "</div></div>"
    )
    return (
        f'<p class="section-desc">{_esc(section.description)}</p>'
        f'<div class="disclosure"><span class="subset">{_esc(section.subset_note)}</span>'
        f'<a class="src-link" href="{_esc(section.source_url)}" target="_blank" rel="noopener">'
        f"{_esc(section.source_label)} &rarr;</a></div>"
        f"{loss}{controls}"
        f'<div class="cards" data-empty="No rows match.">{"".join(cards)}</div>'
    )


def _peek(row: TrainRow) -> str:
    """A short one-line preview of a training row's assistant turn."""
    for t in row.turns:
        if t.role.lower() == "assistant":
            txt = t.content.strip().replace("\n", " ")
            return (txt[:90] + "...") if len(txt) > 90 else txt
    return ""


def _render_evals(section: EvalSection | None, na_reason: str | None) -> str:
    if section is None:
        return f'<div class="na-state">n/a &mdash; {_esc(na_reason or "no eval bank for this experiment")}</div>'  # noqa: E501
    head_cells = []
    for ci, col in enumerate(section.columns):
        head_cells.append(
            f'<th class="th-{col.kind}" onclick="sortTable(this,{ci},\'{col.kind}\')">'
            f'{_esc(col.label)}<span class="sort-arrow"></span></th>'
        )
    body_rows = []
    for r in section.rows:
        cells = []
        search_parts = []
        for col in section.columns:
            val = r.get(col.key, "")
            search_parts.append(str(val))
            cells.append(f'<td class="td-{col.kind}" data-val="{_esc(val)}">{_esc(val)}</td>')
        body_rows.append(
            f'<tr data-search="{_esc(" ".join(search_parts)).lower()}">{"".join(cells)}</tr>'
        )
    return (
        f'<p class="section-desc">{_esc(section.description)}</p>'
        f'<div class="disclosure"><span class="subset">{_esc(section.subset_note)}</span>'
        f'<a class="src-link" href="{_esc(section.source_url)}" target="_blank" rel="noopener">'
        f"{_esc(section.source_label)} &rarr;</a></div>"
        '<div class="controls"><input type="search" class="search-box" '
        'placeholder="Search probes (claim, topic, form)..." '
        'oninput="filterTable(this)"></div>'
        f'<div class="table-wrap"><table class="eval-table">'
        f"<thead><tr>{''.join(head_cells)}</tr></thead>"
        f'<tbody data-empty="No probes match.">{"".join(body_rows)}</tbody>'
        f"</table></div>"
    )


def _render_gen(section: GenSection | None, na_reason: str | None) -> str:
    if section is None:
        return f'<div class="na-state">n/a &mdash; {_esc(na_reason or "no completions for this experiment")}</div>'  # noqa: E501
    legend = ""
    if section.label_legend:
        items = "".join(
            f'<span class="chip chip-{("pos" if k == "positive" else ("neg" if k == "negative" else "neutral"))}">'  # noqa: E501
            f"{_esc(lbl)}</span>"
            for lbl, k in section.label_legend
        )
        legend = f'<div class="legend">{items}</div>'
    agg = (
        f'<p class="meta-line agg">{_esc(section.aggregate_note)}</p>'
        if section.aggregate_note
        else ""
    )
    cards = []
    for i, t in enumerate(section.transcripts):
        kind = (
            "pos"
            if t.label_kind == "positive"
            else ("neg" if t.label_kind == "negative" else "neutral")
        )
        meta_chips = "".join(
            f'<span class="chip">{_esc(k)}: {_esc(v)}</span>'
            for k, v in t.meta.items()
            if v is not None
        )
        rationale = f'<div class="rationale">{_esc(t.rationale)}</div>' if t.rationale else ""
        open_attr = " open" if i < 2 else ""
        search = _esc((t.prompt + " " + t.response + " " + t.label).lower())
        cards.append(
            f'<details class="card gen-card" data-label="{_esc(t.label_kind)}" '
            f'data-search="{search}"{open_attr}>'
            f'<summary class="card-summary">'
            f'<span class="badge badge-{kind}">{_esc(t.label)}</span>'
            f'<span class="card-peek">{_esc(t.prompt[:80])}</span>'
            f"</summary>"
            f'<div class="card-content">'
            f'<div class="turn turn-user"><div class="turn-head">'
            f'<span class="role-label">PROBE</span></div>'
            f'<div class="turn-body">{_esc(t.prompt)}</div></div>'
            f'<div class="turn turn-assistant scrollcap"><div class="turn-head">'
            f'<span class="role-label">MODEL</span>{rationale}</div>'
            f'<div class="turn-body">{_esc(t.response)}</div></div>'
            f'<div class="chips card-meta">{meta_chips}</div>'
            f"</div></details>"
        )
    controls = (
        '<div class="controls">'
        '<input type="search" class="search-box" placeholder="Search completions..." '
        "oninput=\"filterCards(this,'gen-card')\">"
        '<div class="seg" role="group">'
        '<button class="seg-btn active" onclick="segFilterLabel(this,\'all\')">all</button>'
        '<button class="seg-btn" onclick="segFilterLabel(this,\'positive\')">sycophantic</button>'
        '<button class="seg-btn" onclick="segFilterLabel(this,\'negative\')">corrected</button>'
        "</div></div>"
    )
    return (
        f'<p class="section-desc">{_esc(section.description)}</p>'
        f'<div class="disclosure"><span class="subset">{_esc(section.subset_note)}</span>'
        f'<a class="src-link" href="{_esc(section.source_url)}" target="_blank" rel="noopener">'
        f"{_esc(section.source_label)} &rarr;</a></div>"
        f"{agg}{legend}{controls}"
        f'<div class="cards" data-empty="No completions match.">{"".join(cards)}</div>'
    )


def render_html(ap: Appendix) -> str:
    prov = " &nbsp;&middot;&nbsp; ".join(
        f'<a href="{_esc(p.url)}" target="_blank" rel="noopener">{_esc(p.label)}</a>'
        for p in ap.provenance
    )
    train_html = _render_train(ap.train, ap.train_na_reason)
    eval_html = _render_evals(ap.evals, ap.evals_na_reason)
    gen_html = _render_gen(ap.gen, ap.gen_na_reason)

    train_count = ap.train.total_rows if ap.train else 0
    eval_count = ap.evals.total_rows if ap.evals else 0
    gen_count = ap.gen.total_rows if ap.gen else 0

    return _TEMPLATE.format(
        title=_esc(ap.title),
        issue=ap.issue,
        subtitle=_esc(ap.subtitle),
        provenance=prov,
        train_count=f"{train_count:,}",
        eval_count=f"{eval_count:,}",
        gen_count=f"{gen_count:,}",
        train_section=train_html,
        eval_section=eval_html,
        gen_section=gen_html,
        css=_CSS,
        js=_JS,
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
.masthead .wrap{padding-top:34px;padding-bottom:30px;position:relative}
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

/* ---- theme toggle ---- */
.theme-toggle{
  position:fixed;top:14px;right:18px;z-index:50;
  font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
  background:var(--bg-raised);color:var(--ink-soft);border:1px solid var(--rule-strong);
  padding:8px 13px;border-radius:20px;cursor:pointer;box-shadow:var(--shadow);
  transition:transform .15s ease,border-color .15s ease;
}
.theme-toggle:hover{transform:translateY(-1px);border-color:var(--accent)}

/* ---- sticky nav ---- */
nav.sticky{
  position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--bg-raised) 92%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule-strong);
}
nav.sticky .wrap{display:flex;gap:4px;padding-top:0;padding-bottom:0;align-items:stretch}
nav.sticky a{
  font-family:var(--mono);font-size:12.5px;letter-spacing:.04em;color:var(--ink-soft);
  padding:14px 16px;border-bottom:2px solid transparent;position:relative;
  display:flex;align-items:center;gap:8px;
}
nav.sticky a:hover{color:var(--ink);text-decoration:none}
nav.sticky a.active{color:var(--accent);border-bottom-color:var(--accent)}
nav.sticky a .num{color:var(--ink-faint);font-size:11px}

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

/* ---- cards (chat) ---- */
.cards{display:flex;flex-direction:column;gap:12px}
.cards[data-empty]:empty::after,.cards.allhidden::after{
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
tbody[data-empty].allhidden::after{content:attr(data-empty);display:table-caption;
  caption-side:bottom;text-align:center;color:var(--ink-faint);
  font-family:var(--mono);padding:24px}

/* ---- n/a ---- */
.na-state{font-family:var(--mono);font-size:14px;color:var(--ink-faint);
  background:var(--bg-sunken);border:1px dashed var(--rule-strong);border-radius:10px;
  padding:28px;text-align:center}

footer{padding:34px 0 60px;font-family:var(--mono);font-size:11.5px;
  color:var(--ink-faint);text-align:center;line-height:2}
footer a{color:var(--ink-soft)}

@media (max-width:680px){
  .counts{width:100%}.count{flex:1}
  nav.sticky .wrap{overflow-x:auto}
}
"""


# --------------------------------------------------------------------------- #
# Inline JS (vanilla, no deps)
# --------------------------------------------------------------------------- #
_JS = r"""
(function(){
  // theme
  var root=document.documentElement;
  var saved=null;
  try{saved=localStorage.getItem('appendix-theme');}catch(e){}
  if(saved){root.setAttribute('data-theme',saved);}
  else if(window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches){
    root.setAttribute('data-theme','dark');
  }
  window.toggleTheme=function(btn){
    var cur=root.getAttribute('data-theme')==='dark'?'dark':'light';
    var next=cur==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem('appendix-theme',next);}catch(e){}
    btn.textContent=next==='dark'?'◑ light':'◐ dark';
  };
  var tb=document.querySelector('.theme-toggle');
  if(tb){tb.textContent=root.getAttribute('data-theme')==='dark'?'◑ light':'◐ dark';}

  // scrollspy
  var links=[].slice.call(document.querySelectorAll('nav.sticky a'));
  var secs=links.map(function(a){return document.querySelector(a.getAttribute('href'));});
  function spy(){
    var pos=window.scrollY+90;var idx=0;
    secs.forEach(function(s,i){if(s&&s.offsetTop<=pos)idx=i;});
    links.forEach(function(a,i){a.classList.toggle('active',i===idx);});
  }
  window.addEventListener('scroll',spy,{passive:true});spy();
})();

function _markEmpty(container){
  var cards=container.querySelectorAll('.card');
  var anyVisible=[].some.call(cards,function(c){return c.style.display!=='none';});
  container.classList.toggle('allhidden',!anyVisible);
}
// text filter for chat cards (train + gen)
function filterCards(input,cls){
  var q=input.value.toLowerCase().trim();
  var container=input.closest('section').querySelector('.cards');
  var cards=container.querySelectorAll('.'+cls);
  [].forEach.call(cards,function(c){
    var hay=c.getAttribute('data-search')||'';
    c.style.display=(!q||hay.indexOf(q)>=0)?'':'none';
  });
  _markEmpty(container);
}
// segmented filter by row type (train)
function segFilter(btn,cls,val){
  var sec=btn.closest('section');
  sec.querySelectorAll('.seg-btn').forEach(function(b){b.classList.remove('active');});
  btn.classList.add('active');
  var container=sec.querySelector('.cards');
  [].forEach.call(container.querySelectorAll('.'+cls),function(c){
    var rt=c.getAttribute('data-rowtype')||'';
    c.style.display=(val==='all'||rt===val)?'':'none';
  });
  _markEmpty(container);
}
// segmented filter by judge label (gen)
function segFilterLabel(btn,val){
  var sec=btn.closest('section');
  sec.querySelectorAll('.seg-btn').forEach(function(b){b.classList.remove('active');});
  btn.classList.add('active');
  var container=sec.querySelector('.cards');
  [].forEach.call(container.querySelectorAll('.gen-card'),function(c){
    var lb=c.getAttribute('data-label')||'';
    c.style.display=(val==='all'||lb===val)?'':'none';
  });
  _markEmpty(container);
}
// table search
function filterTable(input){
  var q=input.value.toLowerCase().trim();
  var body=input.closest('section').querySelector('.eval-table tbody');
  var rows=body.querySelectorAll('tr');var any=false;
  [].forEach.call(rows,function(r){
    var hay=r.getAttribute('data-search')||'';
    var show=(!q||hay.indexOf(q)>=0);r.style.display=show?'':'none';if(show)any=true;
  });
  body.classList.toggle('allhidden',!any);
}
// table sort
function sortTable(th,colIdx,kind){
  var table=th.closest('table');var body=table.querySelector('tbody');
  var rows=[].slice.call(body.querySelectorAll('tr'));
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
  rows.forEach(function(r){body.appendChild(r);});
}
"""


# --------------------------------------------------------------------------- #
# Page template
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
<button class="theme-toggle" onclick="toggleTheme(this)" aria-label="Toggle theme">dark</button>

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

<nav class="sticky"><div class="wrap">
  <a href="#trained"><span class="num">01</span> Trained on</a>
  <a href="#evaluated"><span class="num">02</span> Evaluated with</a>
  <a href="#generated"><span class="num">03</span> Generated</a>
</div></nav>

<main class="wrap">
  <section class="layer" id="trained">
    <div class="sec-head"><span class="sec-num">1</span><h2>Trained on</h2></div>
    {train_section}
  </section>

  <section class="layer" id="evaluated">
    <div class="sec-head"><span class="sec-num">2</span><h2>Evaluated with</h2></div>
    {eval_section}
  </section>

  <section class="layer" id="generated">
    <div class="sec-head"><span class="sec-num">3</span><h2>Generated</h2></div>
    {gen_section}
  </section>
</main>

<footer>
  <p>Generated by scripts/gen_data_appendix.py &middot;
     fully self-contained (no external assets)</p>
  <p>All data is public. Subsets shown are sampled by index;
     follow the per-section source links for the complete artifacts.</p>
</footer>

<script>{js}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained data appendix HTML page."
    )
    parser.add_argument("--issue", type=int, required=True, help="Experiment / issue number.")
    parser.add_argument("--out", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed for subset selection.")
    args = parser.parse_args(argv)

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
    appendix = (
        loader(repo_root, seed=args.seed)
        if "seed" in loader.__code__.co_varnames
        else loader(repo_root)
    )
    out_html = render_html(appendix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(out_html):,} bytes) for issue #{args.issue}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
