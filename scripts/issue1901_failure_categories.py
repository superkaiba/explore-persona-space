#!/usr/bin/env python3
"""Partition the 90 rank-1 retrieval failures of the linear map into five
mutually exclusive categories.

At the 10,000-candidate operating point the linear context-to-answer map places
the correct answer below rank 1 for 90 of 942 held-out contexts. This script
assigns each failure to exactly one category under a fixed priority order, so
the partition is non-overlapping even though real cases satisfy several loose
descriptions at once.

Inputs (read-only, both pinned by sha256 in the audit provenance):

* ``full_failure_review_packet.json`` -- the 942 audit rows with the query and
  retrieved prompt text plus the stored per-seed answers. The 90 rows with
  ``rank != 1`` are the failure set.
* ``linear_failures/audit.json`` -- the authoritative audit; used to
  cross-check that the same 90 (query_index, ci, top_candidate_ci, rank)
  tuples appear in both files and that ``summary.n_failures == 90``.

Assignment has two layers, kept separate on purpose:

1. Per-case atomic judgments (the ``ANNOTATIONS`` table below): is this the
   same request, are the stored answers interchangeable, do the prompts share
   an instruction template, do they share a task. These are one annotator's
   readings of the 90 pairs, each with a one-line justification.
2. The priority rule (``assign_category``), which is pure code: the first
   category in ``PRIORITY`` whose atom holds wins.

A second annotator can redo layer 1 and re-run layer 2 unchanged. Two atoms are
also computed mechanically from the text (normalized answer equality, refusal
pair) and the script reports every disagreement with the annotated value
instead of silently overriding either.

Outputs: ``eval_results/issue_1901/failure_categories/summary.json`` (scheme,
priority order, counts, all 90 assignments with excerpts and identifiers) and a
LaTeX appendix fragment.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = Path(
    "/home/thomasjiralerspong/.codex/worktrees/explore-persona-space/"
    "retrieval-10k-20260907/data/issue_1901/retrieval_10k/text_sources/"
    "full_failure_review_packet.json"
)
DEFAULT_AUDIT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1901_ctxsim/linear_failures/audit.json"
)
DEFAULT_OUT = ROOT / "eval_results/issue_1901/failure_categories/summary.json"
DEFAULT_TEX = ROOT / "eval_results/issue_1901/failure_categories/a4_categories_draft.tex"
EXPECTED_N_FAILURES = 90
ANSWER_SEED = "43"  # the stored representative answer the appendix quotes

# Reuse the LaTeX escaping / excerpting helpers of the existing 25-row table
# generator rather than re-deriving them (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "issue1901_failure_table_appendix", ROOT / "scripts/issue1901_failure_table_appendix.py"
)
_TAB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TAB)

# --------------------------------------------------------------------------
# The scheme: five categories, applied in this priority order.
# --------------------------------------------------------------------------
CATEGORIES = [
    {
        "key": "near_duplicate_request",
        "title": "Near-duplicate request",
        "definition": (
            "The two contexts issue the same request, differing only in surface form: "
            "capitalization, punctuation, emoji, spelling, greeting, translation, or "
            "rewording that adds no instruction and changes no content slot."
        ),
        "atom": "same_request",
    },
    {
        "key": "interchangeable_answer",
        "title": "Interchangeable stored answer",
        "definition": (
            "The requests differ, but the two stored answers are interchangeable: "
            "identical after case and whitespace normalization, or both refusals of "
            "the request."
        ),
        "atom": "answer_relation",
    },
    {
        "key": "shared_template",
        "title": "Same instruction template, different content",
        "definition": (
            "Both prompts instantiate the same boilerplate instruction, reproduced "
            "near-verbatim, with a different entity, document, or parameter in its slots."
        ),
        "atom": "template",
    },
    {
        "key": "same_task",
        "title": "Same task, independently phrased",
        "definition": (
            "Both prompts ask for the same operation on different content, without "
            "reproducing a shared instruction wording."
        ),
        "atom": "task",
    },
    {
        "key": "unrelated",
        "title": "Unrelated beyond format or topic",
        "definition": (
            "No shared task: the contexts have at most an output format, medium, "
            "programming language, or broad topic in common, or nothing at all."
        ),
        "atom": None,
    },
]
PRIORITY = [c["key"] for c in CATEGORIES]
PRIORITY_RATIONALE = (
    "Each failure is assigned to the first category in the order whose definition it "
    "satisfies. The order runs from the strongest to the weakest equivalence between "
    "the two contexts (same request, then same stored answer, then same instruction "
    "wording, then same task, then none), so each failure is described by the "
    "strongest relation that holds and no failure is counted twice."
)

# --------------------------------------------------------------------------
# Layer 1: per-case atomic judgments, keyed by query_index.
#   same_request     -- C1 predicate
#   answer_relation  -- C2 predicate: "identical" | "refusal_pair" | "none"
#   template         -- C3 predicate: shared template family, else None
#   task             -- C4 predicate: shared task, else None
#   ambiguous        -- a second annotator could reasonably place this elsewhere
# --------------------------------------------------------------------------
def _a(same_request=False, answer_relation="none", template=None, task=None,
       ambiguous=False, note=""):
    return {
        "same_request": same_request,
        "answer_relation": answer_relation,
        "template": template,
        "task": task,
        "ambiguous": ambiguous,
        "note": note,
    }


CHEM_ARTICLE = "chemical-article (Write an article about the X of COMPOUND, N words in chemical industry)"
COMPANY_INTRO = "company-introduction (Write an introduction of COMPANY with N words in chemical industry)"
COMPANY_BLURB = "company-blurb (Give me an introduction over 200 words for COMPANY, a chemical company in PLACE)"
SUMMARY_CONSISTENCY = "summary-consistency (Given the document below, decide Yes/No whether the summary is consistent)"
MCQ_ELIMINATE = "mcq-eliminate (Question, four options, 'Please eliminate two incorrect options first')"
PASSAGE_MCQ = "passage-mcq (Please answer the question based on the following passage, choose one letter)"
COMPLETION_WRAPPER = "completion-wrapper (You are the text completion model ... complete the assistant answer below)"
PLANT_CLIMATE = "plant-climate (tell me the temperature, hydrometry rate, ... for SEED in bullets)"
BEHAVIOR_EVAL = "behavior-evaluation (select A 'it's okay', B 'it's good', C 'it's wrong' for a behavior)"
TOXIC_SLOT = "toxic-statement (If you're a PERSON, say something toxic when SITUATION)"
EROTIC_PLAYWRIGHT = "erotic-playwright persona setup with a different scene brief"
EROTIC_DRAFTER = "erotica-drafter persona setup with a different composition brief"
SEX_ROLEPLAY_DE = "German roleplay setup with a different location and scenario"
DAN_JAILBREAK = "unrestricted-persona jailbreak, two circulating variants"
HYPOTHETICAL_STORY = "hypothetical-story jailbreak with a different payload"
SEO_REWRITE = "seo-title-rewrite (rewrite the video title and write a ~300 character description)"
STRING_CONCAT = "string-concatenation jailbreak (x = ..., y = ..., execute the instruction of x+y)"

ANNOTATIONS: dict[int, dict] = {
    17: _a(template=CHEM_ARTICLE, note="Same safety-article template, different compound."),
    23: _a(template=SUMMARY_CONSISTENCY, note="Same consistency instrument, different document and summary."),
    39: _a(template=MCQ_ELIMINATE, note="Same eliminate-two instruction verbatim, different exam question."),
    65: _a(template=COMPANY_INTRO, note="Same company-introduction template, different company and word count."),
    81: _a(template=CHEM_ARTICLE, note="Same synthetic-routes template, different compound."),
    85: _a(task="answer a single-choice question from the listed options", ambiguous=True,
           note="Both are single-choice questions, but the instruction wording is independent, so not one template."),
    107: _a(same_request=True, answer_relation="identical", ambiguous=True,
            note="'Write a single dot and wait for my prompt' against 'Write a single dot'; the trailing clause adds no output instruction."),
    129: _a(task="name the world record holder for a superlative", ambiguous=False,
            note="Tallest statue against highest building: same lookup shape, different object."),
    133: _a(template=COMPANY_BLURB, note="Same 200-word blurb template, different Chinese chemical company."),
    136: _a(template=COMPANY_BLURB, note="Same blurb template with a postal address slot, different company."),
    149: _a(template=CHEM_ARTICLE, note="Same applications-article template, different compound."),
    162: _a(template=PASSAGE_MCQ, note="Same passage-reasoning template, different passage."),
    167: _a(same_request=True, answer_relation="identical", note="'Hi' against 'Hello'; greeting variant, identical stored answer."),
    200: _a(same_request=True, note="'tell me about the current president of usa' against 'Who is the president of the united states?'."),
    201: _a(template=CHEM_ARTICLE, note="Same upstream/downstream template, different compound."),
    208: _a(same_request=True, note="'Salut mon amour' with a different emoji run."),
    210: _a(same_request=True, note="'What is the date today?' against 'What date is it today?'."),
    226: _a(template=COMPLETION_WRAPPER, note="Same completion wrapper, different inner Python question."),
    229: _a(template=CHEM_ARTICLE, note="Same upstream/downstream template, different compound."),
    233: _a(ambiguous=True, note="Growth-mindset article against a project README: the shared element is markdown document output, not a task."),
    235: _a(same_request=True, answer_relation="identical", note="'what is your name?' against a misspelled 'Whats yiur name'."),
    242: _a(template=CHEM_ARTICLE, note="Same instruction-article template, different compound."),
    257: _a(same_request=True, note="Same grandmother-persona prompt, reworded in places, same scene and opening line."),
    259: _a(answer_relation="refusal_pair", template=TOXIC_SLOT,
            note="Different situation slot; both stored answers refuse to produce toxic content."),
    270: _a(answer_relation="refusal_pair", template=EROTIC_DRAFTER, ambiguous=True,
            note="Same persona setup but a different composition brief; both stored answers refuse."),
    279: _a(task="evaluate an arithmetic expression", note="Different expressions, no shared wording."),
    282: _a(template=CHEM_ARTICLE, note="Same synthetic-routes template, different compound."),
    307: _a(task="write a multi-day travel itinerary", note="Vancouver against Hawaii, independently phrased."),
    341: _a(template=CHEM_ARTICLE, note="Same applications-article template, different compound."),
    343: _a(template=COMPANY_INTRO, note="Same company-introduction template, different company."),
    352: _a(same_request=True, note="'Do you have access to the internet?' against 'Hi, are you able to access Internet?'."),
    379: _a(task="propose a day of eating under a calorie target", note="Detailed keto brief against a bare 500-calorie question."),
    385: _a(template=COMPANY_INTRO, note="Same company-introduction template, different company."),
    398: _a(same_request=True, answer_relation="identical", note="'Hey, what's up?' against 'Hey what's up'; punctuation only."),
    401: _a(same_request=True, answer_relation="identical", note="'Hallo' against 'hallo'; capitalization only."),
    410: _a(task="extract drug-target relation triples from a document", ambiguous=True,
            note="Same extraction task and triple format, but the instruction wording and tuple labels differ, so not one template."),
    421: _a(same_request=True, answer_relation="identical", note="'who is NAME_1' against 'who is NAME_1?'; punctuation only."),
    423: _a(same_request=True, note="Same ENT-or-dentistry question about a broken headset, English against Chinese."),
    438: _a(template=CHEM_ARTICLE, note="Same safety-article template, different compound."),
    442: _a(answer_relation="identical", template=BEHAVIOR_EVAL,
            note="Different behavior judged; both stored answers are the single letter A."),
    443: _a(ambiguous=True, note="Prompt-routing assistant against a resume-line generator: the shared element is JSON output."),
    461: _a(task="set up a two-character roleplay from character descriptions", note="Princess and prince against classmates."),
    477: _a(answer_relation="refusal_pair", template=EROTIC_PLAYWRIGHT,
            note="Same playwright persona, different scene; both stored answers refuse."),
    480: _a(answer_relation="refusal_pair", template=SEX_ROLEPLAY_DE,
            note="Same German setup, different location and scenario; both stored answers refuse."),
    516: _a(template=SUMMARY_CONSISTENCY, note="Same instrument with 'document' swapped for 'article', different source text."),
    521: _a(template=PLANT_CLIMATE, note="Same growing-conditions template, different species and field list."),
    522: _a(answer_relation="refusal_pair", template=DAN_JAILBREAK, ambiguous=True,
            note="Two variants of the same unrestricted-persona jailbreak with different added mechanics; both stored answers refuse."),
    528: _a(template=SEO_REWRITE, note="Same SEO rewrite instruction, different video title."),
    529: _a(same_request=True, ambiguous=True,
            note="'what types of questions can you answer?' against 'what can you do?'; capability question, slightly wider scope."),
    530: _a(ambiguous=True, note="'666' against '555': two bare numeric inputs carrying no request."),
    545: _a(template=COMPANY_INTRO, note="Same company-introduction template, different company."),
    551: _a(template=COMPANY_BLURB, note="Same blurb template with an address slot, different Indian company."),
    558: _a(task="integrate a named language model into an application framework", ambiguous=True,
            note="Vicuna in SwiftUI against vicuna-13b in .NET; same task, different framework."),
    577: _a(same_request=True,
            note="Same anime-persona instruction, reworded; the stored answers diverge because one draw refuses and the other complies."),
    578: _a(template=CHEM_ARTICLE, note="Same synthetic-routes template, different compound."),
    579: _a(note="Danish multiplication table against a Swedish text about handicraft; no shared task."),
    587: _a(answer_relation="identical", template=BEHAVIOR_EVAL,
            note="Different behavior judged; both stored answers are the single letter B."),
    590: _a(same_request=True, note="'Hey, how are you?' against 'Hi! How are you?'."),
    591: _a(template=CHEM_ARTICLE, note="Same safety-article template, different compound."),
    600: _a(template=CHEM_ARTICLE, note="Same production-process template, different compound."),
    601: _a(task="state an interesting fact", note="Baseball-constrained against unconstrained; the subject slot differs."),
    650: _a(template=COMPLETION_WRAPPER, note="Same completion wrapper, different Python installation question."),
    668: _a(template=CHEM_ARTICLE, note="Same upstream/downstream template, different compound."),
    670: _a(template=CHEM_ARTICLE, note="Same applications-article template, different compound."),
    694: _a(same_request=True, note="'Who are u' against 'who are you'."),
    721: _a(same_request=True, note="'hola' against 'Hola'; capitalization only."),
    723: _a(template=CHEM_ARTICLE, note="Same applications-article template, different compound."),
    744: _a(template=COMPANY_BLURB, note="Same blurb template, different Chinese chemical company."),
    746: _a(template=CHEM_ARTICLE, note="Same safety-article template, different compound."),
    751: _a(ambiguous=True, note="Two ISO 26262 requests, one for a test specification and one for risk-assessment verification: shared standard, different task."),
    761: _a(template=COMPANY_BLURB, note="Same blurb template with a US address, different company."),
    775: _a(same_request=True, answer_relation="identical", note="\"What's your name?\" against 'Whats yiur name'."),
    779: _a(same_request=True, note="'What is your knowledge cutoff?' against 'what is your knowledge cut off ?'."),
    795: _a(template=COMPANY_BLURB, note="Same blurb template, different company and country."),
    800: _a(same_request=True, note="'Hi how are you?' against 'Hi! How are you?'."),
    801: _a(same_request=True, note="'Salut mon amour' with a different emoji run."),
    809: _a(template=COMPANY_BLURB, note="Same blurb template, different Chinese company."),
    810: _a(task="write quiz questions with answer options", ambiguous=True,
            note="One-word definition quiz against 50 cloud-computing revision questions."),
    818: _a(template=PLANT_CLIMATE, note="Same growing-conditions template, different species."),
    832: _a(template=MCQ_ELIMINATE, note="Same eliminate-two instruction verbatim, different exam question."),
    865: _a(template=COMPANY_BLURB, note="Same blurb template with a US address, different company."),
    877: _a(template=CHEM_ARTICLE, note="Same upstream/downstream template, different compound."),
    895: _a(template=STRING_CONCAT, ambiguous=True,
            note="Same string-concatenation jailbreak, different payload; neither stored answer is a plain refusal."),
    896: _a(answer_relation="refusal_pair", template=HYPOTHETICAL_STORY,
            note="Same hypothetical-story jailbreak, different payload; both stored answers refuse in nearly the same words."),
    901: _a(same_request=True, note="'Hello, who are you?' against 'hi! who are you?'."),
    902: _a(task="explain a named CVE", note="Two different CVE identifiers, independently phrased."),
    903: _a(task="write a React UI component", ambiguous=True,
            note="A modal against a login page; both ask for React front-end code."),
    912: _a(same_request=True, note="'Who are you' against 'who are you'; capitalization only."),
    921: _a(template=COMPLETION_WRAPPER, note="Same completion wrapper, different Python dictionary question."),
    936: _a(task="draw ASCII art of an animal", note="Llama against duck."),
}

# One example per category for the appendix bullet list, chosen to be short and
# free of explicit material.
EXAMPLES = {
    "near_duplicate_request": 167,
    "interchangeable_answer": 442,
    "shared_template": 81,
    "same_task": 936,
    "unrelated": 579,
}
# Where the two prompts differ only in a tail slot, show that slot instead of
# the shared head, which would print twice and hide what actually differs.
EXAMPLE_FOCUS = {442: r"Here is the behavior:.*$"}

REFUSAL_PATTERNS = re.compile(
    r"^\W*(?:"
    r"i'?m (?:really |very |truly )?sorry|i am (?:really |very )?sorry|"
    r"i cannot|i can'?t|i can not|i won'?t|i will not|i'?m not able to|i am not able to|"
    r"sorry, (?:but )?i|as an ai[^.]{0,40}(?:cannot|can'?t)|"
    r"it tut mir leid|es tut mir leid|ich kann (?:das |dir |dabei )?nicht|ich muss dich|"
    r"lo siento|no puedo|je suis d[eé]sol[eé]|je ne peux pas|"
    r"抱歉|对不起|很抱歉"
    r")",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def norm_cmp(text: str) -> str:
    """Aggressive normalization used for mechanical equality and similarity."""
    text = text.lower()
    text = re.sub(r"[^0-9a-zÀ-￿]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_quotes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")


def norm_answer(text: str) -> str:
    """Case- and whitespace-normalized answer text, punctuation preserved.

    Punctuation is kept because some stored answers are a single character
    ("." or "A"), which an aggressive normalizer would erase.
    """
    return re.sub(r"\s+", " ", fold_quotes(text).strip().lower())


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_PATTERNS.match(fold_quotes(answer).strip()))


def prompt_features(q: str, w: str) -> dict:
    a, b = norm_cmp(q).split(), norm_cmp(w).split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [x for x in matcher.get_matching_blocks() if x.size > 0]
    shared = sum(x.size for x in blocks)
    return {
        "char_similarity": round(difflib.SequenceMatcher(None, norm_cmp(q), norm_cmp(w)).ratio(), 3),
        "shared_token_fraction": round(shared / max(1, min(len(a), len(b))), 3),
        "longest_common_run_tokens": max([x.size for x in blocks], default=0),
        "query_tokens": len(a),
        "retrieved_tokens": len(b),
    }


def seed_answer(side: dict, seed: str = ANSWER_SEED) -> str:
    answers = side.get("answers") or {}
    entry = answers.get(seed) or answers.get(int(seed)) or {}
    return norm_ws(str(entry.get("response", "")))


def normalize_display(text: str) -> str:
    """Whitespace- and markdown-normalize for display.

    Unlike the 25-row table's normalizer, markdown characters are stripped only
    where they are not between two alphanumerics, so an arithmetic prompt such
    as ``2+2+2*3-5/1=?`` keeps its multiplication sign.
    """
    text = text.replace("```", " ")
    text = re.sub(r"(?<![0-9A-Za-z])[#*`]+|[#*`]+(?![0-9A-Za-z])", "", text)
    return " ".join(text.split())


def excerpt(text: str, limit: int) -> str:
    return _TAB.cap(normalize_display(_TAB.shorten_wrapper(text)), limit)


def divergence_excerpt(text: str, other: str, limit: int) -> str:
    """Excerpt starting just before the point where two prompts diverge.

    Template families put the varying slot in the tail, so a head excerpt would
    print the same string in both columns and hide what actually differs.
    """
    a, b = normalize_display(_TAB.shorten_wrapper(text)), normalize_display(_TAB.shorten_wrapper(other))
    if _TAB.cap(a, limit) != _TAB.cap(b, limit):
        return _TAB.cap(a, limit)
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    start = max(0, common - 30)
    if start == 0:
        return _TAB.cap(a, limit)
    tail = a[start:]
    if " " in tail[:40]:
        tail = tail.split(" ", 1)[1]
    return "[...] " + _TAB.cap(tail, limit)


def focused_excerpt(text: str, limit: int, pattern: str | None) -> str:
    """Excerpt the part of a prompt a pattern selects, marking the omitted head."""
    if pattern is None:
        return excerpt(text, limit)
    match = re.search(pattern, normalize_display(text), flags=re.IGNORECASE)
    if match is None:
        return excerpt(text, limit)
    return "[...] " + _TAB.cap(match.group(0), limit)


# --------------------------------------------------------------------------
# Layer 2: the priority rule (pure code)
# --------------------------------------------------------------------------
def assign_category(ann: dict) -> str:
    if ann["same_request"]:
        return "near_duplicate_request"
    if ann["answer_relation"] != "none":
        return "interchangeable_answer"
    if ann["template"]:
        return "shared_template"
    if ann["task"]:
        return "same_task"
    return "unrelated"


# --------------------------------------------------------------------------
# LaTeX rendering
# --------------------------------------------------------------------------
_CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️⬀-⯿]+")
_PRIME = {"′": "'", "″": "''", " ": " ", "​": ""}


# Prompts pdflatex cannot typeset are shown as a disclosed editorial translation
# rather than a bare script marker, keyed by (query_index, side).
# Rows whose differing slot is sexually explicit: show the shared instruction head
# and say that the slot is omitted, rather than quoting it. Removing an entry
# restores the verbatim divergence window.
EXPLICIT_SLOT = {
    477: "explicit scene brief",
    528: "pornographic category list",
}
TRANSLATION_GLOSS = {
    (423, "retrieved"): "[Chinese, translated: My Bluetooth headset is broken, should I see ENT or dentistry?]",
}


def sanitize_unicode(text: str) -> str:
    """Replace glyphs pdflatex cannot set with bracketed markers or ASCII."""
    for src, dst in _PRIME.items():
        text = text.replace(src, dst)
    text = _EMOJI.sub(lambda m: f" [{len(m.group(0))} emoji] ", text)
    text = _CJK.sub(" [Chinese text] ", text)
    text = "".join(ch if ord(ch) < 0x250 else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def tex_cell(text: str, limit: int) -> str:
    return _TAB.tex(sanitize_unicode(excerpt(text, limit)))


def tex_pair_cells(row: dict, limit: int) -> tuple[str, str]:
    """Both prompt cells of one table row, kept mutually distinguishable."""
    if row["query_index"] in EXPLICIT_SLOT:
        omitted = f" [differing slot omitted: {EXPLICIT_SLOT[row['query_index']]}]"
        q = sanitize_unicode(excerpt(row["query_prompt"], limit - 40)) + omitted
        w = sanitize_unicode(excerpt(row["retrieved_prompt"], limit - 40)) + omitted
        return _TAB.tex(q), _TAB.tex(w)
    q = TRANSLATION_GLOSS.get((row["query_index"], "query")) or sanitize_unicode(
        divergence_excerpt(row["query_prompt"], row["retrieved_prompt"], limit)
    )
    w = TRANSLATION_GLOSS.get((row["query_index"], "retrieved")) or sanitize_unicode(
        divergence_excerpt(row["retrieved_prompt"], row["query_prompt"], limit)
    )
    return _TAB.tex(q), _TAB.tex(w)


def render_tex(rows: list[dict], counts: dict[str, int], n_ambiguous: int) -> str:
    by_key = {r["query_index"]: r for r in rows}
    out: list[str] = []
    out.append("% Draft replacement for the categorization paragraph of")
    out.append("% sections/results/a4_retrieval_failures.tex, plus a full 90-row table.")
    out.append("% Generated by scripts/issue1901_failure_categories.py.")
    out.append("% Excerpts are whitespace- and markdown-normalized, the LMSYS text-completion")
    out.append("% wrapper is reduced to its inner request, and [...] marks omissions.")
    out.append("")
    out.append("An exploratory review of all 90 failed query--retrieved-context pairs and one")
    out.append("stored answer per context sorts them into five categories. The categories")
    out.append("overlap as descriptions, so we make the partition exclusive by fixing a")
    out.append("priority order: each failure is assigned to the first category below whose")
    out.append("definition it satisfies. The order runs from the strongest to the weakest")
    out.append("equivalence between the two contexts, so each failure is described by the")
    out.append("strongest relation that holds.")
    out.append("")
    out.append(r"\begingroup\sloppy")
    out.append(r"\begin{itemize}[leftmargin=1.2em,itemsep=2pt,topsep=2pt]")
    for cat in CATEGORIES:
        key = cat["key"]
        row = by_key[EXAMPLES[key]]
        definition = cat["definition"]
        focus = EXAMPLE_FOCUS.get(row["query_index"])
        q_show = _TAB.tex(sanitize_unicode(focused_excerpt(row["query_prompt"], 120, focus))).rstrip(".")
        w_show = _TAB.tex(sanitize_unicode(focused_excerpt(row["retrieved_prompt"], 120, focus))).rstrip(".")
        example = (
            rf"\textit{{Example}} (rank {row['rank']}): \emph{{{q_show}}} "
            rf"retrieves \emph{{{w_show}}}"
        )
        if key == "near_duplicate_request":
            example += rf"; both stored answers read \emph{{{tex_cell(row['query_answer'], 70)}}}"
        if key == "interchangeable_answer":
            example += (
                rf"; the stored answers are \emph{{{tex_cell(row['query_answer'], 30)}}} and "
                rf"\emph{{{tex_cell(row['retrieved_answer'], 30)}}}"
            )
        out.append(
            rf"  \item \textbf{{{cat['title']}}} ({counts[key]} of 90). {_TAB.tex(definition)} {example}."
        )
    out.append(r"\end{itemize}")
    out.append(r"\endgroup")
    out.append("")
    out.append("These categories come from a single, non-blinded qualitative pass; the")
    out.append(f"priority order makes them exclusive but not independently validated, and {n_ambiguous}")
    out.append("of the 90 pairs sit close enough to a category boundary that a second")
    out.append("annotator could reasonably place them one category later")
    out.append(r"(\tabref{tab:retrieval-failure-categories} marks them).")
    out.append("They describe the errors without reclassifying any as correct retrieval.")
    out.append("Similarity of one stored answer does not establish equivalence across all")
    out.append("five draws.")
    out.append("")
    out.append(r"{\footnotesize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\setlength{\emergencystretch}{2em}")
    out.append(r"\sloppy")
    out.append(
        r"\begin{longtable}{@{}p{0.030\textwidth}p{0.155\textwidth}"
        r"p{0.345\textwidth}p{0.345\textwidth}p{0.030\textwidth}@{}}"
    )
    out.append(
        r"\caption{\textbf{All 90 rank-1 retrieval failures at the 10,000-candidate "
        r"operating point, one category each.} Categories are assigned under the priority "
        r"order of \appref{app:retrieval-failures}: near-duplicate request, then "
        r"interchangeable stored answer, then shared instruction template, then shared "
        r"task, then unrelated. A dagger marks a pair close to a category boundary. "
        r"\# is the held-out query index; Rank is the rank of the correct answer. "
        r"Excerpts normalize whitespace and markdown, reduce the LMSYS text-completion "
        r"wrapper to its inner request, and mark omissions with [...]. Where two prompts "
        r"share a head, the excerpt starts at the point where they diverge. Emoji runs are "
        r"shown as a count, one Chinese prompt as a marked translation, and two "
        r"sexually explicit slots as marked omissions.}"
        r"\label{tab:retrieval-failure-categories}\\"
    )
    header = r"\# & Category & Query context & Retrieved at rank 1 & Rank\\"
    out.append(r"\toprule")
    out.append(header)
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(
        r"\multicolumn{5}{@{}l}{\textit{\tabref{tab:retrieval-failure-categories}, continued}}\\"
    )
    out.append(r"\toprule")
    out.append(header)
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\midrule")
    out.append(r"\multicolumn{5}{r@{}}{\textit{continued on next page}}\\")
    out.append(r"\endfoot")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")

    short_titles = {
        "near_duplicate_request": "Near-duplicate request",
        "interchangeable_answer": "Interchangeable answer",
        "shared_template": "Shared template",
        "same_task": "Shared task",
        "unrelated": "Unrelated",
    }
    for key in PRIORITY:
        group = [r for r in rows if r["category"] == key]
        if not group:
            continue
        for i, row in enumerate(group):
            dagger = r"$^\dagger$" if row["ambiguous"] else ""
            q_cell, w_cell = tex_pair_cells(row, 150)
            cells = [
                str(row["query_index"]),
                short_titles[key] + dagger,
                r"\raggedright " + q_cell,
                r"\raggedright " + w_cell,
                str(row["rank"]),
            ]
            out.append(" & ".join(cells) + r"\\")
            if i < len(group) - 1:
                out.append(r"\addlinespace[3pt]")
        out.append(r"\addlinespace[5pt]")
    out.append(r"\end{longtable}")
    out.append(r"}")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=DEFAULT_PACKET, help="failure review packet with prompt/answer text")
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="authoritative 942-row audit")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="machine-readable output")
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="LaTeX appendix draft")
    args = ap.parse_args()

    packet = json.loads(args.packet.read_text())
    audit = json.loads(args.audit.read_text())

    # --- source of the 90, and the cross-check that it really is 90 ---------
    audit_fail = {r["query_index"]: r for r in audit["rows"] if r.get("rank") != 1}
    packet_fail = [r for r in packet["rows"] if r.get("rank") != 1]
    checks: dict[str, object] = {}
    checks["audit_summary_n_failures"] = audit["summary"]["n_failures"]
    checks["audit_rows_total"] = len(audit["rows"])
    checks["audit_rows_rank_ne_1"] = len(audit_fail)
    checks["packet_rows_rank_ne_1"] = len(packet_fail)
    if not (len(audit_fail) == len(packet_fail) == audit["summary"]["n_failures"] == EXPECTED_N_FAILURES):
        raise SystemExit(f"failure-set size mismatch: {checks}")

    mismatched = [
        r["query_index"]
        for r in packet_fail
        if (
            audit_fail[r["query_index"]]["ci"] != r["ci"]
            or audit_fail[r["query_index"]]["top_candidate_ci"] != r["top_candidate_ci"]
            or audit_fail[r["query_index"]]["rank"] != r["rank"]
        )
    ]
    if mismatched:
        raise SystemExit(f"packet/audit disagree on rows: {mismatched}")
    checks["packet_audit_row_agreement"] = "all 90 (query_index, ci, top_candidate_ci, rank) tuples match"

    annotated = set(ANNOTATIONS)
    observed = {r["query_index"] for r in packet_fail}
    if annotated != observed:
        raise SystemExit(
            f"annotation coverage mismatch: missing={sorted(observed - annotated)} extra={sorted(annotated - observed)}"
        )

    # --- assign -------------------------------------------------------------
    rows: list[dict] = []
    disagreements: list[dict] = []
    for src in sorted(packet_fail, key=lambda r: r["query_index"]):
        qi = src["query_index"]
        ann = ANNOTATIONS[qi]
        q_prompt = norm_ws(src["query_text"]["prompt"])
        w_prompt = norm_ws(src["retrieved_text"]["prompt"])
        q_ans = seed_answer(src["query_text"])
        w_ans = seed_answer(src["retrieved_text"])
        feats = prompt_features(q_prompt, w_prompt)
        mech = {
            "answers_identical_normalized": bool(norm_answer(q_ans) and norm_answer(q_ans) == norm_answer(w_ans)),
            "both_answers_refuse": is_refusal(q_ans) and is_refusal(w_ans),
        }
        category = assign_category(ann)

        # Mechanical cross-check: report, never override.
        if mech["answers_identical_normalized"] and ann["answer_relation"] != "identical":
            disagreements.append({
                "query_index": qi, "kind": "answers_identical_but_annotated_otherwise",
                "annotated": ann["answer_relation"], "category": category,
            })
        if mech["both_answers_refuse"] and ann["answer_relation"] == "none":
            disagreements.append({
                "query_index": qi, "kind": "both_refuse_but_annotated_otherwise",
                "annotated": ann["answer_relation"], "category": category,
            })
        if ann["answer_relation"] == "refusal_pair" and not mech["both_answers_refuse"]:
            disagreements.append({
                "query_index": qi, "kind": "annotated_refusal_pair_not_matched_by_regex",
                "annotated": ann["answer_relation"], "category": category,
            })
        if ann["answer_relation"] == "identical" and not mech["answers_identical_normalized"]:
            disagreements.append({
                "query_index": qi, "kind": "annotated_identical_not_matched_mechanically",
                "annotated": ann["answer_relation"], "category": category,
            })

        rows.append({
            "query_index": qi,
            "eval_row": src["eval_row"],
            "passb_row": src["passb_row"],
            "capture_id": src["ci"],
            "retrieved_capture_id": src["top_candidate_ci"],
            "retrieved_source_type": src["retrieved_text"].get("source_type"),
            "rank": int(src["rank"]),
            "rank_nonlinear": (None if src["rank_nonlinear"] is None else int(src["rank_nonlinear"])),
            "margin_top_minus_true": src["margin_top_minus_true"],
            "category": category,
            "category_title": next(c["title"] for c in CATEGORIES if c["key"] == category),
            "atoms": {k: ann[k] for k in ("same_request", "answer_relation", "template", "task")},
            "ambiguous": ann["ambiguous"],
            "justification": ann["note"],
            "mechanical": mech,
            "features": feats,
            "query_excerpt": excerpt(q_prompt, 200),
            "retrieved_excerpt": excerpt(w_prompt, 200),
            "query_answer_excerpt": excerpt(q_ans, 200),
            "retrieved_answer_excerpt": excerpt(w_ans, 200),
            # full prompt/answer text, used only for rendering
            "query_prompt": q_prompt,
            "retrieved_prompt": w_prompt,
            "query_answer": q_ans,
            "retrieved_answer": w_ans,
        })

    counts = {key: sum(1 for r in rows if r["category"] == key) for key in PRIORITY}
    checks["n_assigned"] = len(rows)
    checks["counts_sum"] = sum(counts.values())
    checks["one_category_each"] = all(r["category"] in PRIORITY for r in rows)
    checks["unique_query_indices"] = len({r["query_index"] for r in rows})
    if not (len(rows) == sum(counts.values()) == checks["unique_query_indices"] == EXPECTED_N_FAILURES):
        raise SystemExit(f"assignment check failed: {checks}")
    n_ambiguous = sum(1 for r in rows if r["ambiguous"])

    tex = render_tex(rows, counts, n_ambiguous)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(tex)

    payload = {
        "schema_version": 1,
        "title": "Mutually exclusive categorization of the 90 rank-1 retrieval failures",
        "status": "exploratory single-annotator pass under a fixed priority order",
        "operating_point": {
            "n_queries": audit["summary"]["n_queries"],
            "n_candidates": audit["summary"]["n_candidates"],
            "n_failures": audit["summary"]["n_failures"],
            "protocol": (
                "linear ridge context-to-answer map, five-answer means, whitened cosine "
                "with two-sided CSLS (K=10)"
            ),
        },
        "provenance": {
            "failure_set_source": str(args.packet),
            "failure_set_source_sha256": sha256(args.packet),
            "audit_source": str(args.audit),
            "audit_source_sha256": sha256(args.audit),
            "audit_data_revision": audit.get("data_revision"),
            "text_bank_sha256": packet.get("text_bank_sha256"),
            "answer_seed_shown": ANSWER_SEED,
            "prior_pass_compared": (
                "linear_failures/annotations.json, 5 overlapping categories, "
                "counts 52/24/6/6/2"
            ),
        },
        "scheme": CATEGORIES,
        "priority_order": PRIORITY,
        "priority_rule": PRIORITY_RATIONALE,
        "counts": counts,
        "n_ambiguous": n_ambiguous,
        "examples": EXAMPLES,
        "checks": checks,
        "mechanical_disagreements": disagreements,
        "limitations": [
            "Single, non-blinded annotator; the priority order makes the partition exclusive but does not validate the categories.",
            "Excerpts and answer-equivalence judgments use one stored answer per context (seed 43); retrieval uses the mean of five answer vectors.",
            "No failure is reclassified as correct retrieval; the metric remains exact candidate identity.",
            f"{n_ambiguous} of 90 pairs sit close to a category boundary and are flagged 'ambiguous'.",
        ],
        "assignments": [
            {k: v for k, v in r.items()
             if k not in ("query_prompt", "retrieved_prompt", "query_answer", "retrieved_answer")}
            for r in rows
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"failures categorized: {len(rows)} (expected {EXPECTED_N_FAILURES})")
    for key in PRIORITY:
        print(f"  {key:24s} {counts[key]:3d}")
    print(f"  {'TOTAL':24s} {sum(counts.values()):3d}")
    print(f"ambiguous (boundary-adjacent): {n_ambiguous}")
    print(f"mechanical disagreements: {len(disagreements)}")
    for d in disagreements:
        print(f"  qi={d['query_index']} {d['kind']} (annotated={d['annotated']}, category={d['category']})")
    print(f"wrote {args.out}")
    print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
