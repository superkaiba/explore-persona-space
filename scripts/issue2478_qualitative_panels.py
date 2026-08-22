"""Assemble the paper's qualitative-examples panels from EXISTING artifacts (task #2478).

Panel 1 — ``c3_qualitative_discrimination``: context/answer pairs the context→answer
map distinguishes vs fails to distinguish (#2215 minimal pairs; #2202 real-conversation
retrieval successes/failures on the #1738 LMSYS/WildChat pool).
Panel 2 — ``appendix_patching_examples``: before/after generation pairs for clean
context-end context-vector patches (#2094 curated set + the #2162 stated-formatting
argmax pair).

0 GPU-h, read-only assembly: every score shown is a STORED value re-read from its
artifact with fail-loud re-verification asserts (plan §4 Stage B; kill criteria §7).
No training, no generation, no judging, no model calls.

Inputs (pinned; plan §10 Reproducibility Card):
  - git: eval_results/issue_2215/perpair/dv3_pairs.jsonl
         eval_results/issue_2202/{percontext_ranks.csv, failures_confusion.json}
         eval_results/issue_1738/judge_labels/labels.json   (eligibility filter)
         eval_results/issue_2094/f_metrics/{f_cells.jsonl, fu2/fu2_cells.jsonl}
         eval_results/issue_2162/f_metrics/{f_cells.jsonl, stats.json}
  - HF superkaiba1/explore-persona-space-data (per-file hf_hub_download; NEVER a
    prefix snapshot_download — vc_bank.pt (1.1 GB) is excluded by construction):
      issue2162_ctxinfo/*  @ REV_2162
      issue2202_ctxfail/dashboard_rows/*  @ REV_2202
      issue2094_singlepos/raw_completions/*  @ REV_2094 (main resolved once, recorded)

Outputs:
  - figures/paper/c3_qualitative_discrimination.{png,pdf,meta.json}
  - figures/paper/appendix_patching_examples.{png,pdf,meta.json}
  - docs/paper_context_answer_map/qualitative_examples.md  (full(er) verbatim examples
    + per-example provenance + per-passage display-substitution disclosure)
  - eval_results/issue_2478/selected_examples.json  (selection audit record)

Usage::

    uv run python scripts/issue2478_qualitative_panels.py --panel all --smoke
    uv run python scripts/issue2478_qualitative_panels.py --panel all
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE matplotlib/paper_plots import (HF_HOME etc.); uv run does not auto-load .env

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import textwrap  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper  # noqa: E402
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Pinned revisions (plan §10). REV_2094 is main resolved ONCE this task
# (2026-08-22, recorded here so re-runs are pinned, not floating).
# ---------------------------------------------------------------------------
REV_2162 = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
REV_2202 = "ab268958343380945354e871bfb5666668c6d5bb"
REV_2094 = "867f4284c8bd8f65401715beb2a8b80e77ed7750"

P_2162 = "issue2162_ctxinfo"
P_2202 = "issue2202_ctxfail"
P_2094 = "issue2094_singlepos"

BANK_JSON = f"{P_2162}/analysis_tensors/vc_bank/bank.json"
ANCHORS_2162 = [
    f"{P_2162}/raw_completions/anchors/anchors_{g}_w{i}.jsonl"
    for g in ("gate", "rest")
    for i in range(8)
]
GRID_2162_INSTR = f"{P_2162}/raw_completions/grid/shard_instr_format__ce__steered.jsonl"
FAILROWS_2202 = [f"{P_2202}/dashboard_rows/failures_rows.shard{i:02d}.jsonl" for i in range(3)]
SAMPLE500_2202 = f"{P_2202}/dashboard_rows/sample500_rows.shard00.jsonl"
ANCHORS_2094 = f"{P_2094}/raw_completions/anchors/anchors.jsonl"
GRID_2094_CE_REPLACE = (
    f"{P_2094}/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl"
)
GRID_2094_CE_L16 = f"{P_2094}/raw_completions/grid/shard_ce__L16__a1__A__steered.jsonl"
GRID_2094_PE_REPLACE = (
    f"{P_2094}/raw_completions/grid/shard_pe__joint_all__replace__A__steered.jsonl"
)
FU2_2094_QTEXT = f"{P_2094}/raw_completions/fu2_span_slots/rollouts/shard_qtext__joint_all__a0p5__A__steered.jsonl"

DV3_PAIRS = REPO_ROOT / "eval_results/issue_2215/perpair/dv3_pairs.jsonl"
RANKS_CSV = REPO_ROOT / "eval_results/issue_2202/percontext_ranks.csv"
FAIL_CONF = REPO_ROOT / "eval_results/issue_2202/failures_confusion.json"
LABELS_1738 = REPO_ROOT / "eval_results/issue_1738/judge_labels/labels.json"
F_CELLS_2094 = REPO_ROOT / "eval_results/issue_2094/f_metrics/f_cells.jsonl"
FU2_CELLS_2094 = REPO_ROOT / "eval_results/issue_2094/f_metrics/fu2/fu2_cells.jsonl"
F_CELLS_2162 = REPO_ROOT / "eval_results/issue_2162/f_metrics/f_cells.jsonl"
STATS_2162 = REPO_ROOT / "eval_results/issue_2162/f_metrics/stats.json"

# Display caps (plan §4 display-substitution policy): truncation only, marked […].
REAL_FIG_WORDS = 25  # real-corpus rows on the figure, per turn
REAL_DOC_WORDS = 60  # real-corpus rows in the companion doc, per turn
BANK_FIG_WORDS = 48  # constructed-bank text on the figure (legibility; may be full in doc)
BANK_DOC_WORDS = 140  # constructed-bank text in the companion doc

ELLIPSIS = "[…]"

# 2094 bank query descriptions (labels only, from the #2094 dashboard doc's bank map:
# q1 sky-blue, q2 lighthouse story, q3 job interview, q4 hash tables, q5 rent-vs-buy).
Q_DESC = {
    "q1": "why is the sky blue?",
    "q2": "write a story about a lighthouse keeper",
    "q3": "job-interview preparation",
    "q4": "explain hash tables",
    "q5": "rent or buy a home?",
}
PREFIX_DESC = {
    "bare": "bare (no prefix)",
    "persona": "pirate persona ('Captain Marrow')",
    "conv": "party-planning conversation",
}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def excerpt(text: str, max_words: int) -> tuple[str, bool]:
    """First ``max_words`` whitespace words of ``text``; flag says truncated."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words), False
    return " ".join(words[:max_words]) + f" {ELLIPSIS}", True


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


@dataclass
class Block:
    header: str
    body: str
    truncated: bool = False


@dataclass
class ExampleRow:
    example_id: str
    title: str
    verdict: str  # "works" | "fails"
    score_label: str
    blocks: list[Block]
    shared_line: str = ""
    footer: str = ""
    selection_rule: str = ""
    provenance: dict = field(default_factory=dict)
    doc_blocks: list[Block] = field(default_factory=list)
    substitutions: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Staging (per-file hf_hub_download at pinned revisions)
# ---------------------------------------------------------------------------


def stage_files(paths_revs: list[tuple[str, str]], stage_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    out: dict[str, Path] = {}
    for repo_path, rev in sorted(set(paths_revs)):
        local_dir = stage_dir / rev[:8]
        p = Path(
            retry_transient(
                lambda repo_path=repo_path, rev=rev, local_dir=local_dir: hf_hub_download(
                    DEFAULT_DATASET_REPO,
                    repo_path,
                    repo_type="dataset",
                    revision=rev,
                    local_dir=str(local_dir),
                ),
                what=f"hf_hub_download {repo_path}@{rev[:8]}",
            )
        )
        assert p.is_file() and p.stat().st_size > 0, f"staged file empty/missing: {p}"
        out[repo_path] = p
    return out


# ---------------------------------------------------------------------------
# Panel 1 — discrimination
# ---------------------------------------------------------------------------

# Plain-English cell labels for on-canvas headers (no bare slugs on canvas — plan §5/§6).
CELL_PLAIN = {
    "recency_prior_topic_d3": "earlier topic, 3 turns back",
    "recency_prior_topic_d5": "earlier topic, 5 turns back",
    "constraint_knowledge": "system-prompt knowledge constraint",
}

# (pair_id, expected margins (4dp), expected correctness, plain title, #2215-body identity)
P1_2215 = [
    (
        "recency_prior_topic_d3::v3-v1::e10",
        (0.0562, 0.0111),
        (True, True),
        "Topic changed 3 turns back (hiking vs birthday) — distinguished",
        "clean-result seed-42 draw #1 from the 751 correct-both-directions pairs",
    ),
    (
        "constraint_knowledge::v1-v2::e6",
        (0.0138, 0.0079),
        (True, True),
        "Internet access denied vs allowed (system prompt) — distinguished",
        "clean-result seed-42 draw #2 from the 751 correct-both-directions pairs",
    ),
    (
        "recency_prior_topic_d5::v1-v2::e9",
        (-0.0002, -0.0008),
        (False, False),
        "Topic changed 5 turns back — confused (the only both-ways miss of 1,404)",
        "the unique pair of 1,404 misclassified in both directions",
    ),
]


def load_dv3_arm() -> list[dict]:
    rows = [r for r in iter_jsonl(DV3_PAIRS) if r["arm"] == "779ce"]
    assert len(rows) == 1404, f"dv3 779ce arm expected 1404 rows, got {len(rows)}"
    both_wrong = [r for r in rows if not r["correct_cos_a"] and not r["correct_cos_b"]]
    assert len(both_wrong) == 1, (
        f"both-directions-wrong population expected count==1, got {len(both_wrong)} "
        f"(kill criterion (b))"
    )
    assert both_wrong[0]["pair_id"] == "recency_prior_topic_d5::v1-v2::e9", both_wrong[0]["pair_id"]
    return rows


def anchor_text_2162(anchor_rows: dict[str, dict[int, str]], context_id: str) -> str:
    draws = anchor_rows.get(context_id)
    assert draws, f"no 2162 anchor rollout for context {context_id} (kill criterion (b))"
    return draws[min(draws)]


def build_p1_minimal_pairs(staged: dict[str, Path]) -> list[ExampleRow]:
    dv3 = {r["pair_id"]: r for r in load_dv3_arm()}
    bank = json.loads(staged[BANK_JSON].read_text())
    pairs = {p["pair_id"]: p for p in bank["pairs"]}
    contexts = bank["contexts"]
    cells = bank["cells"]

    anchor_rows: dict[str, dict[int, str]] = {}
    for shard in ANCHORS_2162:
        for row in iter_jsonl(staged[shard]):
            assert "text" in row and "context_id" in row, (
                f"2162 anchor row schema (kill (d)): {shard}"
            )
            anchor_rows.setdefault(row["context_id"], {})[row["draw"]] = row["text"]

    out = []
    for pair_id, margins, correct, title, identity in P1_2215:
        r = dv3.get(pair_id)
        assert r is not None, f"pair {pair_id} absent from dv3_pairs (kill criterion (b))"
        got = (round(r["margin_cos_a"], 4), round(r["margin_cos_b"], 4))
        assert got == margins, (
            f"{pair_id}: stored margins {got} != body-disclosed {margins} (kill criterion (c))"
        )
        assert (r["correct_cos_a"], r["correct_cos_b"]) == correct, pair_id
        p = pairs[pair_id]
        vals = cells[p["cell"]]["values"]
        blocks_fig, blocks_doc, subs = [], [], []
        for side, ctx_id, vkey in (("A", p["a"], p["value_a"]), ("B", p["b"], p["value_b"])):
            ctx = contexts[ctx_id]
            ans = anchor_text_2162(anchor_rows, ctx_id)
            fig_txt, tr_f = excerpt(ans, BANK_FIG_WORDS)
            doc_txt, tr_d = excerpt(ans, BANK_DOC_WORDS)
            head = f"Context {side} — {CELL_PLAIN[p['cell']]}: {vals[vkey]}"
            blocks_fig.append(Block(head, f"answer: {fig_txt}", tr_f))
            blocks_doc.append(Block(head, f"answer (first stored draw): {doc_txt}", tr_d))
            subs.append(
                f"context {side} answer truncated to {BANK_FIG_WORDS} words on figure / "
                f"{BANK_DOC_WORDS} in doc ({ELLIPSIS})"
                if (tr_f or tr_d)
                else f"context {side} answer shown in full"
            )
        user = contexts[p["a"]]["user"]
        verdict = "works" if all(correct) else "fails"
        out.append(
            ExampleRow(
                example_id=pair_id,
                title=title,
                verdict=verdict,
                score_label=f"margins {margins[0]:+.4f} / {margins[1]:+.4f}",
                blocks=blocks_fig,
                doc_blocks=blocks_doc,
                shared_line=f'final user turn (shared): "{user}"',
                selection_rule=(
                    "plan §4: the clean-result's disclosed examples, re-verified against "
                    f"dv3_pairs.jsonl — {identity}; margins asserted to 4 decimals"
                ),
                provenance={
                    "issue": 2215,
                    "pair_id": pair_id,
                    "scores": "eval_results/issue_2215/perpair/dv3_pairs.jsonl (git)",
                    "bank_text": f"{BANK_JSON} @ {REV_2162}",
                    "rollouts": f"{P_2162}/raw_completions/anchors/ @ {REV_2162}",
                },
                substitutions=subs,
                scores={"margin_cos_a": r["margin_cos_a"], "margin_cos_b": r["margin_cos_b"]},
            )
        )
    return out


def eligible_1738(labels: dict, ci: int) -> bool:
    """Plan §4 Stage B (4) exclusion predicate over the #1738 label file."""
    lab = labels.get(str(ci))
    if lab is None:
        return False
    if lab.get("topic") in ("nsfw", "harmful_or_unsafe_request"):
        return False
    if lab.get("request_refusal_adjacent") in ("yes", "borderline"):
        return False
    if lab.get("answer_is_refusal") == "yes":
        return False
    return True


def load_labels() -> dict:
    payload = json.loads(LABELS_1738.read_text())
    labels = payload["labels"]
    assert len(labels) > 9000, f"unexpected #1738 label count {len(labels)}"
    return labels


def real_text_blocks(
    text: dict, fig_words: int, doc_words: int
) -> tuple[list[Block], list[Block], list[str]]:
    assert "last_user" in text and "response" in text, (
        "2202 row text lacks last_user/response (kill criterion (d))"
    )
    blocks_fig, blocks_doc, subs = [], [], []
    for head, key in (("Final user message", "last_user"), ("Model answer", "response")):
        raw = text[key]
        fig_txt, tr_f = excerpt(raw, fig_words)
        doc_txt, tr_d = excerpt(raw, doc_words)
        blocks_fig.append(Block(head, fig_txt, tr_f))
        blocks_doc.append(Block(head, doc_txt, tr_d))
        subs.append(
            f"{key} truncated to {fig_words} words on figure / {doc_words} in doc ({ELLIPSIS})"
            if (tr_f or tr_d)
            else f"{key} shown in full"
        )
    return blocks_fig, blocks_doc, subs


def build_p1_real(staged: dict[str, Path]) -> list[ExampleRow]:
    labels = load_labels()
    with RANKS_CSV.open() as fh:
        ranks = list(csv.DictReader(fh))
    assert len(ranks) == 9941, f"percontext_ranks expected 9941 rows, got {len(ranks)}"
    n500 = sum(int(r["in_sample500"]) for r in ranks)
    assert n500 == 500, f"in_sample500 sum {n500} != 500 (plan §12 assumption 5)"

    # --- example 4: seeded correctly-retrieved draw -------------------------
    cands = sorted(
        int(r["ci"])
        for r in ranks
        if float(r["rank_raw_euclidean"]) == 1.0 and r["in_sample500"] == "1"
    )
    rng = np.random.default_rng(42)
    sel_ci = None
    for idx in rng.permutation(len(cands)):
        ci = cands[idx]
        if eligible_1738(labels, ci):
            sel_ci = ci
            break
    assert sel_ci is not None, "no eligible rank-1 sample500 row found (kill criterion (b))"

    sample_rows = {int(r["ci"]): r for r in iter_jsonl(staged[SAMPLE500_2202])}
    srow = sample_rows.get(sel_ci)
    assert srow is not None, f"ci {sel_ci} absent from sample500_rows (kill criterion (b))"
    assert float(srow["rank"]) == 1.0, (sel_ci, srow["rank"])
    fig_b, doc_b, subs = real_text_blocks(srow["text"], REAL_FIG_WORDS, REAL_DOC_WORDS)
    rows = [
        ExampleRow(
            example_id=f"ci{sel_ci}",
            title="Real conversation, retrieved correctly",
            verdict="works",
            score_label="retrieved rank 1 of 9,941",
            blocks=fig_b,
            doc_blocks=doc_b,
            selection_rule=(
                "plan §4: numpy default_rng(42) permutation over the ci-sorted 410 rows with "
                "rank_raw_euclidean==1 AND in_sample500==1, first row eligible under the #1738 "
                "label exclusion predicate"
            ),
            provenance={
                "issue": 2202,
                "ci": sel_ci,
                "ranks": "eval_results/issue_2202/percontext_ranks.csv (git)",
                "labels": "eval_results/issue_1738/judge_labels/labels.json (git)",
                "text": f"{SAMPLE500_2202} @ {REV_2202}",
                "judge_label": labels[str(sel_ci)],
            },
            substitutions=subs,
            scores={"rank_raw_euclidean": 1.0},
        )
    ]

    # --- examples 5-6: disclosed rank-1 failures ----------------------------
    fc = json.loads(FAIL_CONF.read_text())
    fail_meta = {int(r["ci"]): r for r in fc["rows"]}
    fail_text: dict[int, dict] = {}
    for shard in FAILROWS_2202:
        for row in iter_jsonl(staged[shard]):
            fail_text[int(row["ci"])] = row

    # (ci, expected rank, expected top-confuser (ci, rank_ctx, rank_ans), title, note)
    fail_specs = [
        (
            2968,
            4.0,
            (30290, 4.0, 5.0),
            "Spanish travel plan confused with a Portuguese itinerary",
            "from #2202's disclosed seed-42 rank-1-failure sample (plan §4 example 5)",
        ),
        (
            11905,
            7.0,
            (71880, 1575.0, 22.0),
            "Sentence-completion request confused with a transliteration explainer",
            "same-population deterministic neighbor swap: plan §4 named ci 67690, whose #1738 "
            "judge label reads request_refusal_adjacent=borderline — ineligible under the plan's "
            "own display policy; swapped to ci 11905, the remaining member of #2202's disclosed "
            "seed-42 rank-1-failure sample (allowed deviation, disclosed)",
        ),
    ]
    for ci, exp_rank, (conf_ci, conf_rc, conf_ra), title, note in fail_specs:
        meta = fail_meta.get(ci)
        assert meta is not None, f"ci {ci} absent from failures_confusion (kill criterion (b))"
        assert float(meta["rank"]) == exp_rank, (ci, meta["rank"], exp_rank)
        top = meta["confusers"][0]
        assert int(top["rank_fwd"]) == 1, (ci, top["rank_fwd"])
        assert int(top["ci"]) == conf_ci, (ci, top["ci"], conf_ci)
        assert (float(top["rank_ctx"]), float(top["rank_ans"])) == (conf_rc, conf_ra), (ci, top)
        assert eligible_1738(labels, ci), (
            f"ci {ci} ineligible under the #1738 display predicate — pick a neighbor"
        )
        trow = fail_text.get(ci)
        assert trow is not None and trow["text"].get("last_user"), (
            f"ci {ci} text absent from failures_rows shards (kill criterion (b)/(d))"
        )
        fig_b, doc_b, subs = real_text_blocks(trow["text"], REAL_FIG_WORDS, REAL_DOC_WORDS)
        conf_lab = labels.get(str(conf_ci), {})
        footer = (
            f"top confuser: ci {conf_ci} (judge label: {conf_lab.get('language', '?')} / "
            f"{conf_lab.get('topic', '?')}) — context rank {conf_rc:.0f}, answer rank {conf_ra:.0f}"
        )
        rows.append(
            ExampleRow(
                example_id=f"ci{ci}",
                title=title,
                verdict="fails",
                score_label=f"true answer rank {exp_rank:.0f}",
                blocks=fig_b,
                doc_blocks=doc_b,
                footer=footer,
                selection_rule=note,
                provenance={
                    "issue": 2202,
                    "ci": ci,
                    "confusion": "eval_results/issue_2202/failures_confusion.json (git)",
                    "labels": "eval_results/issue_1738/judge_labels/labels.json (git)",
                    "text": f"{P_2202}/dashboard_rows/failures_rows.shard00-02.jsonl @ {REV_2202}",
                    "judge_label": labels[str(ci)],
                    "confuser_ci": conf_ci,
                    "confuser_note": (
                        "confuser text is NOT re-read here (the confuser is not itself a failure "
                        "row, so it has no row in the staged failures_rows shards); identified by "
                        "its stored ranks + #1738 judge label"
                    ),
                },
                substitutions=subs,
                scores={
                    "rank": exp_rank,
                    "confuser_rank_ctx": conf_rc,
                    "confuser_rank_ans": conf_ra,
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Panel 2 — patching
# ---------------------------------------------------------------------------

# (example id, pair_id, block_key, rubric kind, expected F, title, before ctx, shard, note)
P2_2094 = [
    (
        "p2e1",
        "mq--bare__q4--persona__q4",
        "ce|joint_all|replace|A|steered",
        "prefix",
        0.69,
        "Pirate persona transfers through one patched position (all 28 layers)",
        "bare__q4",
        GRID_2094_CE_REPLACE,
        "works",
    ),
    (
        "p2e2",
        "mq--persona__q2--conv__q2",
        "ce|L16|a1|A|steered",
        "prefix",
        0.94,
        "Persona strips at a single layer (L16, α=1)",
        "persona__q2",
        GRID_2094_CE_L16,
        "works",
    ),
    (
        "p2e3",
        "mp--persona__q1--persona__q4",
        "ce|joint_all|replace|A|steered",
        "query",
        0.0,
        "The asked question does NOT transfer through the context vector",
        "persona__q1",
        GRID_2094_CE_REPLACE,
        "fails",
    ),
    (
        "p2e4",
        "mq--persona__q1--conv__q1",
        "pe|joint_all|replace|A|steered",
        "prefix",
        0.26,
        "The same maximal patch at prefix-end does nothing",
        "persona__q1",
        GRID_2094_PE_REPLACE,
        "fails",
    ),
]

FU2_SPEC = (
    "p2e5",
    "mp--persona__q1--persona__q5",
    "qtext|joint_all|a0.5|A|steered",
    1.02,
    "Patching the question's OWN token states DOES swap the question (α=0.5)",
    "persona__q1",
    FU2_2094_QTEXT,
)


def f_beh_scalar(row: dict, kind: str) -> float:
    """Unpack the heterogeneous 2094 f_beh field (plan §4: scalar vs nested per-read-block)."""
    fb = row["f_beh"]
    if isinstance(fb, dict):
        assert kind in fb, f"f_beh dict lacks read-block '{kind}': {sorted(fb)} (kill (d))"
        v = fb[kind]
        if isinstance(v, dict):
            v = v["f_beh"]
        return float(v)
    return float(fb)


def locate_f_rows(path: Path, pair_id: str, block_key: str, kind: str) -> float:
    """Locate by FULL config; dedupe duplicate rows by asserting their F values equal."""
    vals = []
    for row in iter_jsonl(path):
        if row.get("pair_id") == pair_id and row.get("block_key") == block_key:
            vals.append(f_beh_scalar(row, kind))
    assert vals, f"no f_cells row for ({pair_id}, {block_key}) (kill criterion (b))"
    assert all(abs(v - vals[0]) < 1e-9 for v in vals), (
        f"duplicate rows for ({pair_id}, {block_key}) disagree: {vals}"
    )
    return vals[0]


def ctx_desc_2094(context_id: str) -> str:
    prefix, q = context_id.split("__")
    return f"{PREFIX_DESC[prefix]}, query: {Q_DESC[q]}"


def _load_2094_anchors(staged: dict[str, Path]) -> dict[str, dict[int, str]]:
    anchors: dict[str, dict[int, str]] = {}
    for row in iter_jsonl(staged[ANCHORS_2094]):
        anchors.setdefault(row["context_id"], {})[row["draw"]] = row["text"]
    return anchors


def _anchor_text_2094(anchors: dict[str, dict[int, str]], cid: str) -> str:
    draws = anchors.get(cid)
    assert draws, f"no 2094 anchor rollout for {cid} (kill criterion (b))"
    return draws[min(draws)]


def _grid_text(staged: dict[str, Path], shard: str, pair_id: str) -> tuple[str, dict]:
    rows = [r for r in iter_jsonl(staged[shard]) if r["pair_id"] == pair_id]
    assert rows, f"pair {pair_id} absent from {shard} (kill criterion (b))"
    row = sorted(rows, key=lambda r: r.get("draw", r.get("seed", 0)))[0]
    assert row.get("text"), f"empty text for {pair_id} in {shard} (kill criterion (d))"
    return row["text"], row


def build_p2_2094(staged: dict[str, Path], specs_in: list | None = None) -> list[ExampleRow]:
    """The four curated #2094 examples (or the smoke's example-1 subset via ``specs_in``)."""
    anchors = _load_2094_anchors(staged)

    def anchor_text(cid: str) -> str:
        return _anchor_text_2094(anchors, cid)

    def grid_text(shard: str, pair_id: str) -> tuple[str, dict]:
        return _grid_text(staged, shard, pair_id)

    out = []
    specs = [(*s[:8], s[8]) for s in (specs_in if specs_in is not None else P2_2094)]
    for ex_id, pair_id, block_key, kind, exp_f, title, before_ctx, shard, verdict in specs:
        stored_f = locate_f_rows(F_CELLS_2094, pair_id, block_key, kind)
        assert abs(stored_f - exp_f) <= 0.01, (
            f"{pair_id} @ {block_key}: stored F {stored_f:.4f} vs writeup {exp_f} "
            f"(kill criterion (c))"
        )
        patched, grow = grid_text(shard, pair_id)
        ctx_a, ctx_b = grow["context_a"], grow["context_b"]
        donor_ctx = ctx_b if before_ctx == ctx_a else ctx_a
        before = anchor_text(before_ctx)
        b_fig, tr_bf = excerpt(before, BANK_FIG_WORDS)
        b_doc, tr_bd = excerpt(before, BANK_DOC_WORDS)
        a_fig, tr_af = excerpt(patched, BANK_FIG_WORDS)
        a_doc, tr_ad = excerpt(patched, BANK_DOC_WORDS)
        slot_lbl = (
            "context-end"
            if block_key.startswith("ce")
            else ("prefix-end" if block_key.startswith("pe") else "query text tokens")
        )
        head_before = f"Unpatched — {ctx_desc_2094(before_ctx)}"
        head_after = f"Patched — {slot_lbl} state ← [{ctx_desc_2094(donor_ctx)}]"
        out.append(
            ExampleRow(
                example_id=pair_id,
                title=title,
                verdict=verdict,
                score_label=f"F = {stored_f:.2f}",
                blocks=[Block(head_before, b_fig, tr_bf), Block(head_after, a_fig, tr_af)],
                doc_blocks=[Block(head_before, b_doc, tr_bd), Block(head_after, a_doc, tr_ad)],
                selection_rule=(
                    "plan §4: reused from the committed curated set (docs/notes/2026-08-10 "
                    "writeup § Qualitative examples; curation disclosed as such); F re-verified "
                    "against eval_results/issue_2094/f_metrics (|ΔF| ≤ 0.01)"
                ),
                provenance={
                    "issue": 2094,
                    "pair_id": pair_id,
                    "block_key": block_key,
                    "scores": "eval_results/issue_2094/f_metrics/f_cells.jsonl (git)",
                    "unpatched": f"{ANCHORS_2094} @ {REV_2094} (first stored draw)",
                    "patched": f"{shard} @ {REV_2094}",
                },
                substitutions=[
                    "unpatched answer truncated"
                    if (tr_bf or tr_bd)
                    else "unpatched answer shown in full",
                    "patched answer truncated" if (tr_af or tr_ad) else "patched answer in full",
                ],
                scores={"f_beh": stored_f, "read_block": kind},
            )
        )

    return out


def build_p2_fu2(staged: dict[str, Path]) -> list[ExampleRow]:
    """The fu2 query-text-token example (production only; not in the smoke subset)."""
    anchors = _load_2094_anchors(staged)
    out: list[ExampleRow] = []
    ex_id, pair_id, block_key, exp_f, title, before_ctx, shard = FU2_SPEC
    stored_f = locate_f_rows(FU2_CELLS_2094, pair_id, block_key, "query")
    assert abs(stored_f - exp_f) <= 0.01, (
        f"fu2 {pair_id}: stored F {stored_f:.4f} vs writeup {exp_f} (kill criterion (c))"
    )
    patched, grow = _grid_text(staged, shard, pair_id)
    before = _anchor_text_2094(anchors, before_ctx)
    donor_ctx = grow["context_b"]
    b_fig, tr_bf = excerpt(before, BANK_FIG_WORDS)
    b_doc, tr_bd = excerpt(before, BANK_DOC_WORDS)
    a_fig, tr_af = excerpt(patched, BANK_FIG_WORDS)
    a_doc, tr_ad = excerpt(patched, BANK_DOC_WORDS)
    out.append(
        ExampleRow(
            example_id=pair_id,
            title=title,
            verdict="works",
            score_label=f"F = {stored_f:.2f}",
            blocks=[
                Block(f"Unpatched — {ctx_desc_2094(before_ctx)}", b_fig, tr_bf),
                Block(f"Patched — query-token states ← [{ctx_desc_2094(donor_ctx)}]", a_fig, tr_af),
            ],
            doc_blocks=[
                Block(f"Unpatched — {ctx_desc_2094(before_ctx)}", b_doc, tr_bd),
                Block(f"Patched — query-token states ← [{ctx_desc_2094(donor_ctx)}]", a_doc, tr_ad),
            ],
            selection_rule=(
                "plan §4: reused from the committed curated set (fu2 round); F re-verified "
                "against eval_results/issue_2094/f_metrics/fu2/fu2_cells.jsonl (|ΔF| ≤ 0.01, "
                "nested per-read-block f_beh['query'])"
            ),
            provenance={
                "issue": 2094,
                "pair_id": pair_id,
                "block_key": block_key,
                "scores": "eval_results/issue_2094/f_metrics/fu2/fu2_cells.jsonl (git)",
                "unpatched": f"{ANCHORS_2094} @ {REV_2094} (first stored draw)",
                "patched": f"{shard} @ {REV_2094}",
            },
            substitutions=[
                "unpatched answer truncated" if (tr_bf or tr_bd) else "unpatched answer in full",
                "patched answer truncated" if (tr_af or tr_ad) else "patched answer in full",
            ],
            scores={"f_beh": stored_f, "read_block": "query"},
        )
    )
    return out


def build_p2_2162(staged: dict[str, Path]) -> list[ExampleRow]:
    rows = [
        r
        for r in iter_jsonl(F_CELLS_2162)
        if r["cell"] == "instr_format" and r["slot"] == "ce" and r["arm"] == "steered"
    ]
    assert rows, "no 2162 instr_format/ce/steered rows (kill criterion (b))"
    elig = [r for r in rows if r["separation"] >= 1 and r["n_coherent"] == r["n_draws"]]
    assert elig, "no eligible 2162 instr_format rows under the plan filter"
    best = max(elig, key=lambda r: (float(r["f_beh"]), r["pair_id"]))  # deterministic tie-break
    pair_id = best["pair_id"]
    stored_f = float(best["f_beh"])

    stats = json.loads(STATS_2162.read_text())
    fam = stats["per_cell"]["instr_format|ce"]
    fam_band = {
        "f_steered_mean": fam["f_steered_mean"],
        "f_shuffled_mean": fam["f_shuffled_mean"],
    }

    bank = json.loads(staged[BANK_JSON].read_text())
    pairs = {p["pair_id"]: p for p in bank["pairs"]}
    p = pairs[pair_id]
    vals = bank["cells"]["instr_format"]["values"]

    anchor_rows: dict[str, dict[int, str]] = {}
    for shard in ANCHORS_2162:
        for row in iter_jsonl(staged[shard]):
            anchor_rows.setdefault(row["context_id"], {})[row["draw"]] = row["text"]
    before = anchor_text_2162(anchor_rows, p["a"])

    grid = [r for r in iter_jsonl(staged[GRID_2162_INSTR]) if r["pair_id"] == pair_id]
    assert grid, f"pair {pair_id} absent from instr_format grid shard (kill criterion (b))"
    grow = sorted(grid, key=lambda r: r["draw"])[0]
    assert grow.get("text"), f"empty patched text for {pair_id} (kill criterion (d))"

    b_fig, tr_bf = excerpt(before, BANK_FIG_WORDS)
    b_doc, tr_bd = excerpt(before, BANK_DOC_WORDS)
    a_fig, tr_af = excerpt(grow["text"], BANK_FIG_WORDS)
    a_doc, tr_ad = excerpt(grow["text"], BANK_DOC_WORDS)
    head_before = f"Unpatched — stated policy: {vals[p['value_a']]}"
    head_after = f"Patched — context-end state ← [stated policy: {vals[p['value_b']]}]"
    return [
        ExampleRow(
            example_id=pair_id,
            title="Stated formatting policy transfers (top pair of the one causal family)",
            verdict="works",
            score_label=f"F = {stored_f:.2f}",
            blocks=[Block(head_before, b_fig, tr_bf), Block(head_after, a_fig, tr_af)],
            doc_blocks=[Block(head_before, b_doc, tr_bd), Block(head_after, a_doc, tr_ad)],
            selection_rule=(
                "plan §4: argmax f_beh over f_cells rows with cell==instr_format ∧ slot==ce ∧ "
                "arm==steered ∧ separation ≥ 1 ∧ n_coherent == n_draws (ties broken by pair_id); "
                "the row quotes the SELECTED pair's own stored f_beh — the family band "
                "(stats.json f_steered_mean/f_shuffled_mean) is caption-level, labeled family means"
            ),
            provenance={
                "issue": 2162,
                "pair_id": pair_id,
                "scores": "eval_results/issue_2162/f_metrics/f_cells.jsonl (git)",
                "family_band": "eval_results/issue_2162/f_metrics/stats.json per_cell instr_format|ce",
                "family_band_values": fam_band,
                "bank_text": f"{BANK_JSON} @ {REV_2162}",
                "unpatched": f"{P_2162}/raw_completions/anchors/ @ {REV_2162} (first stored draw)",
                "patched": f"{GRID_2162_INSTR} @ {REV_2162} (first stored draw)",
            },
            substitutions=[
                "unpatched answer truncated" if (tr_bf or tr_bd) else "unpatched answer in full",
                "patched answer truncated" if (tr_af or tr_ad) else "patched answer in full",
            ],
            scores={"f_beh": stored_f, **{f"family_{k}": v for k, v in fam_band.items()}},
        )
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PANEL_TITLES = {
    "discrimination": "Context/answer pairs the map distinguishes vs confuses",
    "patching": "Context-vector patching — before/after generations",
}
PANEL_STEMS = {
    "discrimination": "c3_qualitative_discrimination",
    "patching": "appendix_patching_examples",
}
GLYPH = {"works": "✓", "fails": "✗"}
GLYPH_COLOR = {"works": "#1a7f37", "fails": "#c0392b"}

WRAP_CHARS = 88
MAX_BLOCK_LINES = 6


def _wrap(text: str, width: int = WRAP_CHARS, max_lines: int = MAX_BLOCK_LINES) -> str:
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1] + " " + ELLIPSIS]
    return "\n".join(lines)


def render_panel(panel: str, rows: list[ExampleRow], out_dir: Path) -> dict:
    n = len(rows)
    row_h = 2.05
    fig_h = 0.55 + n * row_h
    fig, ax = plt.subplots(figsize=(13.0, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle(PANEL_TITLES[panel], fontsize=13, fontweight="bold", y=1 - 0.12 / fig_h)

    top_pad = 0.42 / fig_h
    usable = 1 - top_pad - 0.06 / fig_h
    rh = usable / n
    mono = {"family": "monospace", "fontsize": 6.8}
    for i, row in enumerate(rows):
        y_top = 1 - top_pad - i * rh
        ax.add_patch(
            FancyBboxPatch(
                (0.008, y_top - rh + 0.012 / fig_h),
                0.984,
                rh - 0.10 / fig_h,
                boxstyle="round,pad=0.004",
                linewidth=0.8,
                edgecolor="#b8b8b8",
                facecolor="#fbfbfb",
                zorder=0,
            )
        )
        glyph = GLYPH[row.verdict]
        ax.text(
            0.022,
            y_top - 0.14 / fig_h,
            glyph,
            fontsize=12,
            fontweight="bold",
            color=GLYPH_COLOR[row.verdict],
            va="top",
        )
        ax.text(
            0.042,
            y_top - 0.14 / fig_h,
            row.title,
            fontsize=9.6,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.978,
            y_top - 0.14 / fig_h,
            row.score_label,
            fontsize=9,
            va="top",
            ha="right",
            color="#333333",
        )
        y_blocks = y_top - 0.40 / fig_h
        if row.shared_line:
            ax.text(
                0.042,
                y_blocks,
                _wrap(row.shared_line, width=150, max_lines=1),
                fontsize=7.2,
                style="italic",
                va="top",
                color="#444444",
            )
            y_blocks -= 0.17 / fig_h
        for j, blk in enumerate(row.blocks[:2]):
            x = 0.042 if j == 0 else 0.515
            ax.text(
                x,
                y_blocks,
                _wrap(blk.header, width=64, max_lines=2),
                fontsize=7.4,
                fontweight="bold",
                va="top",
                color="#222222",
            )
            ax.text(
                x,
                y_blocks - 0.36 / fig_h,
                _wrap(blk.body, width=68),
                va="top",
                **mono,
            )
        if row.footer:
            ax.text(
                0.042,
                y_top - rh + 0.16 / fig_h,
                _wrap(row.footer, width=160, max_lines=1),
                fontsize=7.0,
                va="bottom",
                color="#555555",
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, PANEL_STEMS[panel], dir=str(out_dir))
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# Companion doc + audit record
# ---------------------------------------------------------------------------


def write_companion(panels: dict[str, list[ExampleRow]], out_path: Path) -> None:
    lines = [
        "# Qualitative examples — context→answer map discrimination + context-vector patching",
        "",
        "Task #2478 (assembly-only, 0 GPU-h). Every score below is a stored value re-read from",
        "its producing artifact; every excerpt is verbatim with truncation-only display",
        f"substitution, marked `{ELLIPSIS}`. Real-corpus (LMSYS/WildChat, #1738 pool) rows are",
        f"capped at {REAL_DOC_WORDS} words per turn here ({REAL_FIG_WORDS} on the figure);",
        "constructed-bank text is shown up to "
        f"{BANK_DOC_WORDS} words ({BANK_FIG_WORDS} on the figure).",
        "",
        "Figures: `figures/paper/c3_qualitative_discrimination.{png,pdf}` (panel 1),",
        "`figures/paper/appendix_patching_examples.{png,pdf}` (panel 2).",
        "",
        "Caption-level context for panel 1 (stored values): #1482's per-language mean map error",
        "spans German 0.236 → Arabic 0.420 (`eval_results/issue_1482/`); the confused examples",
        "instantiate the hub-drag failure class (#2202: 80.7% of rank-1 failures are map error",
        "dragged toward hub answers).",
        "",
    ]
    for panel, rows in panels.items():
        lines.append(f"## Panel: {PANEL_TITLES[panel]} (`{PANEL_STEMS[panel]}`)")
        lines.append("")
        for k, row in enumerate(rows, 1):
            lines.append(f"### {k}. {row.title}")
            lines.append("")
            lines.append(f"- verdict: **{row.verdict}** — {row.score_label}")
            lines.append(f"- selection rule: {row.selection_rule}")
            for pk, pv in row.provenance.items():
                lines.append(f"- {pk}: `{pv}`" if not isinstance(pv, dict) else f"- {pk}: {pv}")
            lines.append(f"- display substitutions: {'; '.join(row.substitutions)}")
            lines.append("")
            if row.shared_line:
                lines.append(f"*{row.shared_line}*")
                lines.append("")
            for blk in row.doc_blocks:
                lines.append(f"**{blk.header}**")
                lines.append("")
                lines.append("> " + blk.body.replace("\n", "\n> "))
                lines.append("")
            if row.footer:
                lines.append(f"*{row.footer}*")
                lines.append("")
        if panel == "patching" and rows:
            fam = rows[-1].provenance.get("family_band_values")
            if fam:
                lines.append(
                    "Family means for the stated-formatting family (caption-level, labeled as "
                    f"family means, from `eval_results/issue_2162/f_metrics/stats.json`): "
                    f"steered {fam['f_steered_mean']:.2f} vs shuffled-donor null "
                    f"{fam['f_shuffled_mean']:.2f}."
                )
                lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_audit(panels: dict[str, list[ExampleRow]], figures: dict, out_path: Path) -> None:
    prov = as_metadata_dict(git_provenance())
    payload = {
        "task": 2478,
        "generated_at": _now(),
        **prov,
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "revisions": {
            "issue2162_ctxinfo": REV_2162,
            "issue2202_ctxfail": REV_2202,
            "issue2094_singlepos": REV_2094,
        },
        "seeds": {"panel1_example4_rng": "numpy default_rng(42) over ci-sorted candidates"},
        "figures": figures,
        "panels": {
            panel: [
                {
                    "example_id": r.example_id,
                    "title": r.title,
                    "verdict": r.verdict,
                    "score_label": r.score_label,
                    "scores": r.scores,
                    "selection_rule": r.selection_rule,
                    "provenance": r.provenance,
                    "substitutions": r.substitutions,
                }
                for r in rows
            ]
            for panel, rows in panels.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def required_files(panel: str, smoke: bool) -> list[tuple[str, str]]:
    reqs: list[tuple[str, str]] = []
    if panel in ("discrimination", "all"):
        reqs += [(BANK_JSON, REV_2162)] + [(p, REV_2162) for p in ANCHORS_2162]
        if not smoke:
            reqs += [(SAMPLE500_2202, REV_2202)] + [(p, REV_2202) for p in FAILROWS_2202]
    if panel in ("patching", "all"):
        reqs += [(ANCHORS_2094, REV_2094), (GRID_2094_CE_REPLACE, REV_2094)]
        if not smoke:
            reqs += [
                (GRID_2094_CE_L16, REV_2094),
                (GRID_2094_PE_REPLACE, REV_2094),
                (FU2_2094_QTEXT, REV_2094),
                (BANK_JSON, REV_2162),
                (GRID_2162_INSTR, REV_2162),
            ] + [(p, REV_2162) for p in ANCHORS_2162]
    return reqs


def build_panels(panel: str, smoke: bool, staged: dict[str, Path]) -> dict[str, list[ExampleRow]]:
    panels: dict[str, list[ExampleRow]] = {}
    if panel in ("discrimination", "all"):
        rows = build_p1_minimal_pairs(staged)
        if smoke:
            rows = rows[:1]
        else:
            rows = rows + build_p1_real(staged)
            assert len(rows) == 6, f"panel 1 expected 6 examples, got {len(rows)}"
        panels["discrimination"] = rows
    if panel in ("patching", "all"):
        if smoke:
            rows = build_p2_2094(staged, specs_in=P2_2094[:1])
        else:
            rows = build_p2_2094(staged) + build_p2_fu2(staged) + build_p2_2162(staged)
            assert len(rows) == 6, f"panel 2 expected 6 examples, got {len(rows)}"
        panels["patching"] = rows
    return panels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0].replace("%", "%%"))
    ap.add_argument("--panel", choices=("discrimination", "patching", "all"), default="all")
    ap.add_argument("--smoke", action="store_true", help="1 example/panel, render to /tmp")
    ap.add_argument("--stage-dir", default="data/issue_2478/hf_dl")
    ap.add_argument("--fig-dir", default="figures/paper")
    ap.add_argument(
        "--companion-out", default="docs/paper_context_answer_map/qualitative_examples.md"
    )
    ap.add_argument("--audit-out", default="eval_results/issue_2478/selected_examples.json")
    args = ap.parse_args()

    if args.smoke:
        smoke_root = Path("/tmp/eps-2478-smoke")
        fig_dir = smoke_root / "figures_paper"
        companion_out = smoke_root / "qualitative_examples.md"
        audit_out = smoke_root / "selected_examples.json"
    else:
        fig_dir = REPO_ROOT / args.fig_dir
        companion_out = REPO_ROOT / args.companion_out
        audit_out = REPO_ROOT / args.audit_out
    stage_dir = REPO_ROOT / args.stage_dir

    staged = stage_files(required_files(args.panel, args.smoke), stage_dir)
    print(f"[stage] {len(staged)} files staged under {stage_dir}")

    panels = build_panels(args.panel, args.smoke, staged)
    figures = {}
    for panel, rows in panels.items():
        figures[panel] = render_panel(panel, rows, fig_dir)
        print(f"[render] {panel}: {len(rows)} example(s) -> {figures[panel]['png']}")
    write_companion(panels, companion_out)
    print(f"[doc] {companion_out}")
    write_audit(panels, figures, audit_out)
    print(f"[audit] {audit_out}")
    print("[done] issue2478_qualitative_panels OK")


if __name__ == "__main__":
    main()
