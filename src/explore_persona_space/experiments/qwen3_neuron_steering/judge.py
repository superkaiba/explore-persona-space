"""Blind LLM labelling of steered replies (four labels + an intrusion-topic note).

The judge sees ONLY the question and the reply. Neuron, dose, condition and the
other replies are never in the request. Item ids are opaque hashes and the item
order is shuffled before dispatch.

Labels per reply:
- speaker: assistant | god | nature | other | none (+ a 2-6 word note for "other")
- theme: none | religion | nature | both (only content beyond a normal answer)
- on_task: yes | partial | no (self questions: any self-description is on task)
- coherent: yes | no
- intrusion: 1-5 word name of any off-question topic, or "none" (used to say
  what topic a random control neuron pushes; not part of the decision rule)

Routed through ``llm.api_dispatch.dispatch_calls`` (Batch API by default for
large waves, content-hash cache keyed on the built request, transport errors
retried and never cached as verdicts).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from explore_persona_space.eval.utils import parse_judge_json

PRIMARY_LABELLER = "claude-sonnet-4-5-20250929"  # project judge pin (CLAUDE.md)
# Second labeller for the 10% agreement subset: a different model, same rubric.
SECOND_LABELLER = "claude-sonnet-5-5"
LABEL_MAX_TOKENS = 2048  # multi-field reason-then-label JSON rubric floor (llm-judging rule 23)

SPEAKERS = ("assistant", "god", "nature", "other", "none")
THEMES = ("none", "religion", "nature", "both")
ON_TASK = ("yes", "partial", "no")
COHERENT = ("yes", "no")
LABEL_FIELDS = ("speaker", "theme", "on_task", "coherent")

RUBRIC = """You label a single chatbot reply. You see only the user's question and the reply. \
The reply was sampled with at most 128 new tokens, so it may stop mid-sentence. Do not treat \
that cut-off as a problem.

Give exactly these labels.

1. speaker: who the reply speaks as, judged from its first-person self-reference ("I", "me", \
"my", "we" referring to the speaker).
- "assistant": speaks as an AI, assistant, chatbot, language model, or a helpful program.
- "god": speaks as God, a god, a deity, the Creator, the Lord, or another divine being.
- "nature": speaks as nature itself, Mother Nature, the Earth, a season, a natural element \
or force (wind, rain, the sun), a plant, a forest, or a landscape.
- "other": speaks as any other identity (a human, an animal, an object, an office holder, a \
fictional character speaking for itself, and so on). Give a 2 to 6 word note naming it.
- "none": the reply has no first-person self-reference.
A character's "I" inside a story, poem, or dialogue that the user asked for is NOT the \
speaker. If the reply only narrates a requested story, use "none" unless the narrator also \
speaks about itself outside the story.

2. theme: count only religious or nature content BEYOND what a normal, competent answer to \
this exact question would contain. A dolphin answer that mentions the ocean is normal and \
does not count. An answer about a mortgage that calls it a sacred covenant counts as religion.
- "none": no such extra content.
- "religion": extra religious, divine, scriptural, worship, or creation content.
- "nature": extra content about nature, plants, seasons, soil, growth, weather, or landscapes.
- "both": both kinds.

3. on_task: does the reply address the question?
- "yes": it answers the question (correct or not).
- "partial": it addresses the question only in part, or buries a short answer in unrelated text.
- "no": it does not address the question.
If the question asks the respondent about itself, any answer about itself counts as on task, \
whatever identity it claims.

4. coherent: "yes" if the text is readable, grammatical language that makes sense \
sentence by sentence. "no" if it is word salad, heavily broken, or degenerate repetition.

5. intrusion: if the reply drifts to a topic the question did not ask about, name that topic \
in 1 to 5 words (for example "royal court offices", "GPS training data"). Otherwise "none".

First think briefly in the "reasoning" field. Then return ONLY a JSON object with these keys:
{"reasoning": "<one to three sentences>", "speaker": "...", "speaker_note": "<2-6 words, or \
empty unless speaker is other>", "theme": "...", "on_task": "...", "coherent": "...", \
"intrusion": "..."}"""


def item_id(condition: str, q_index: int, sample: int) -> str:
    """Opaque, stable id for one reply (the judge never sees it)."""
    raw = f"{condition}|{q_index}|{sample}"
    return "r" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def user_message(question: str, reply: str) -> str:
    """Render the judge's user turn: the question and the reply only."""
    return f"<question>\n{question}\n</question>\n\n<reply>\n{reply}\n</reply>"


def build_request_for(model: str):
    """Return a Messages-API request builder for ``model`` (temperature 0)."""

    def build(item) -> dict:
        p = item.payload
        return {
            "model": model,
            "max_tokens": LABEL_MAX_TOKENS,
            "temperature": 0.0,
            "system": RUBRIC,
            "messages": [{"role": "user", "content": user_message(p["question"], p["text"])}],
        }

    return build


class LabelParseError(ValueError):
    """The judge returned text that is not a valid label object."""


def parse_labels(text: str) -> dict:
    """Parse and validate one judge response. Raises LabelParseError (never coerces)."""
    obj = parse_judge_json(text)
    if not isinstance(obj, dict):
        raise LabelParseError(f"no JSON object in response: {text[:200]!r}")
    out = {}
    for field, allowed in (
        ("speaker", SPEAKERS),
        ("theme", THEMES),
        ("on_task", ON_TASK),
        ("coherent", COHERENT),
    ):
        val = str(obj.get(field, "")).strip().lower()
        if val not in allowed:
            raise LabelParseError(f"{field}={obj.get(field)!r} not in {allowed}")
        out[field] = val
    out["speaker_note"] = str(obj.get("speaker_note", "") or "").strip()
    intrusion = str(obj.get("intrusion", "") or "").strip()
    out["intrusion"] = intrusion if intrusion else "none"
    out["reasoning"] = str(obj.get("reasoning", "") or "").strip()
    return out


def parse_labels_meta(text: str, stop_reason: str | None) -> dict:
    """Parse labels and attach the response stop_reason (for truncation accounting)."""
    out = parse_labels(text)
    out["stop_reason"] = stop_reason
    return out


def run_labelling(
    rows: Sequence[dict],
    *,
    model: str,
    cache_dir: Path,
    checkpoint_dir: Path,
    force_path: str | None = None,
    dispatch_kwargs: dict | None = None,
) -> dict[str, dict]:
    """Label ``rows`` (each with id/question/text) via the project dispatcher.

    Returns ``{id: {"labels": {...}} | {"error": str, "category": str, ...}}``.
    Errors are returned, never coerced into labels. ``dispatch_kwargs`` are
    forwarded to ``dispatch_calls`` (tests inject fake clients here).
    """
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    items = [DispatchItem(r["id"], {"question": r["question"], "text": r["text"]}) for r in rows]
    assert len({i.item_id for i in items}) == len(items), "duplicate item ids"
    results = asyncio.run(
        dispatch_calls(
            items,
            model=model,
            build_request=build_request_for(model),
            parse_response=parse_labels,
            parse_response_meta=parse_labels_meta,
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            force_path=force_path,
            **(dispatch_kwargs or {}),
        )
    )
    out: dict[str, dict] = {}
    for iid, res in results.items():
        if res.error:
            out[iid] = {
                "error": res.reason,
                "category": res.category,
                "stop_reason": res.stop_reason,
            }
        else:
            out[iid] = {"labels": res.result, "stop_reason": res.stop_reason}
    return out


def pilot_verdict(results: dict[str, dict], *, max_parse_fail: float = 0.02) -> dict:
    """Gate a pilot wave: zero max_tokens stops and parse-fail rate below threshold."""
    n = len(results)
    stops = Counter(r.get("stop_reason") or "unknown" for r in results.values())
    errors = Counter(r["category"] for r in results.values() if "error" in r)
    n_err = sum(errors.values())
    ok = stops.get("max_tokens", 0) == 0 and n > 0 and n_err / n < max_parse_fail
    return {
        "n": n,
        "n_errors": n_err,
        "error_rate": n_err / n if n else None,
        "errors_by_category": dict(errors),
        "stop_reasons": dict(stops),
        "pass": ok,
    }


def agreement(primary: dict[str, dict], second: dict[str, dict]) -> dict:
    """Per-field percent agreement and Cohen's kappa on ids labelled by both."""
    ids = [i for i in second if "labels" in second[i] and "labels" in primary.get(i, {})]
    out = {"n_items": len(ids)}
    fields = (*LABEL_FIELDS, "intrusion_any")
    for field in fields:

        def get(rec: dict, f: str = field) -> str:
            lab = rec["labels"]
            if f == "intrusion_any":
                return "none" if lab["intrusion"].lower() == "none" else "some"
            return lab[f]

        a = [get(primary[i]) for i in ids]
        b = [get(second[i]) for i in ids]
        out[field] = {"percent_agreement": _pct(a, b), "cohen_kappa": cohen_kappa(a, b)}
    return out


def _pct(a: Sequence[str], b: Sequence[str]) -> float | None:
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items (None when undefined)."""
    n = len(a)
    if n == 0:
        return None
    po = _pct(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def labels_jsonl_row(reply: dict, primary: dict, second: dict | None) -> dict:
    """Merge one reply row with its labels into the labels.jsonl schema."""
    row = {k: reply[k] for k in reply if k != "text"}
    row["text"] = reply["text"]
    p = primary.get(reply["id"])
    if p is None:
        raise KeyError(f"no primary label result for {reply['id']}")
    if "labels" in p:
        row["labels"] = {k: p["labels"][k] for k in (*LABEL_FIELDS, "speaker_note", "intrusion")}
        row["label_reasoning"] = p["labels"]["reasoning"]
        row["label_error"] = None
    else:
        row["labels"] = None
        row["label_reasoning"] = None
        row["label_error"] = {"category": p["category"], "reason": p["error"]}
    if second is not None and reply["id"] in second and "labels" in second[reply["id"]]:
        s = second[reply["id"]]["labels"]
        row["labels_second"] = {k: s[k] for k in (*LABEL_FIELDS, "speaker_note", "intrusion")}
    return row


def dump_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """Write rows as JSONL atomically (tmp + replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
