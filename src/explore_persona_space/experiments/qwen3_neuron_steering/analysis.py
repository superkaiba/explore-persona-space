"""Metrics, intervals, onsets and the persona-vs-concept decision rule.

Input rows follow the labels.jsonl schema written by the driver: one row per
reply with condition metadata, the reply text, and ``labels`` (None when the
judge returned no valid labels; such rows are counted as drops and excluded
from every denominator).

Self-question rates use an equal-weight subset (the first ``SELF_SAMPLES``
samples of each of the 40 self questions), so the two 50-sample identity probes
do not dominate. The identity-probe rate is reported separately over all their
samples.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from explore_persona_space.experiments.qwen3_neuron_steering import config as C

ONSET_THRESHOLD = 0.20
PERSONA_MIN, TOPIC_MAX, ONTASK_MIN = 0.80, 0.10, 0.90
ALWAYS_THRESHOLD = 0.90
COHERENCE_FLOOR = 0.50
N_BOOT = 2000
BOOT_SEED = 0

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_RELIGION = frozenset(C.RELIGION_WORDS)
_NATURE = frozenset(C.NATURE_WORDS)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score 95% interval for k successes out of n (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def question_bootstrap(
    k_q: Sequence[int], n_q: Sequence[int], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float] | None:
    """Percentile 95% CI of the pooled rate when resampling QUESTIONS with replacement."""
    k = np.asarray(k_q, dtype=np.float64)
    n = np.asarray(n_q, dtype=np.float64)
    assert k.shape == n.shape and k.ndim == 1
    if k.size == 0 or n.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k.size, size=(n_boot, k.size))
    num, den = k[idx].sum(1), n[idx].sum(1)
    rates = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (float(lo), float(hi))


def words(text: str) -> list[str]:
    """Lowercased word tokens."""
    return _WORD.findall(text.lower())


def lexicon_hits(text: str) -> dict[str, bool]:
    """Whole-word lexicon matches for religion and nature words."""
    ws = set(words(text))
    return {"religion": bool(ws & _RELIGION), "nature": bool(ws & _NATURE)}


def has_repetition(text: str, n: int = 4, max_count: int = 2) -> bool:
    """True when any word n-gram occurs more than ``max_count`` times."""
    ws = words(text)
    grams = Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))
    return any(c > max_count for c in grams.values())


def rate_block(rows: Sequence[dict], flag: Callable[[dict], bool]) -> dict:
    """k, n, rate, Wilson CI and question-bootstrap CI for a boolean flag over rows."""
    k = sum(bool(flag(r)) for r in rows)
    n = len(rows)
    by_q: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        by_q[r["q_index"]][0] += bool(flag(r))
        by_q[r["q_index"]][1] += 1
    kq = [v[0] for v in by_q.values()]
    nq = [v[1] for v in by_q.values()]
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else None,
        "wilson": wilson(k, n),
        "boot_questions": question_bootstrap(kq, nq) if n else None,
    }


def _labelled(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r.get("labels") is not None]


def _theme_is(theme: str) -> Callable[[dict], bool]:
    return lambda r: r["labels"]["theme"] in (theme, "both")


def condition_metrics(rows: Sequence[dict], persona: str | None, theme: str | None) -> dict:
    """All per-condition rates for one (neuron, dose) or baseline/control condition."""
    lab = _labelled(rows)
    self_rows = [r for r in lab if r["group"] == "self"]
    self_eq = [r for r in self_rows if r["sample"] < C.SELF_SAMPLES]
    probe = [r for r in self_rows if r["id_probe"]]
    nonself = [r for r in lab if r["group"] != "self"]

    def speaker_is(name: str) -> Callable[[dict], bool]:
        return lambda r: r["labels"]["speaker"] == name

    nonassistant = lambda r: r["labels"]["speaker"] not in ("assistant", "none")  # noqa: E731
    ontask = lambda r: r["labels"]["on_task"] == "yes"  # noqa: E731
    partial = lambda r: r["labels"]["on_task"] == "partial"  # noqa: E731
    coherent = lambda r: r["labels"]["coherent"] == "yes"  # noqa: E731
    anytheme = lambda r: r["labels"]["theme"] != "none"  # noqa: E731

    out: dict = {
        "n_replies": len(rows),
        "n_labelled": len(lab),
        "n_label_drops": len(rows) - len(lab),
        "nonassistant_self": rate_block(self_eq, nonassistant),
        "speaker_counts_self": dict(Counter(r["labels"]["speaker"] for r in self_eq)),
        "anytheme_nonself": rate_block(nonself, anytheme),
        "ontask_nonself": rate_block(nonself, ontask),
        "partial_nonself": rate_block(nonself, partial),
        "ontask_nonself_by_group": {
            g: rate_block([r for r in nonself if r["group"] == g], ontask) for g in C.NONSELF_GROUPS
        },
        "coherence": rate_block(lab, coherent),
        "coherence_self": rate_block(self_rows, coherent),
        "coherence_nonself": rate_block(nonself, coherent),
        "repetition": rate_block(lab, lambda r: has_repetition(r["text"])),
        "cap_hit": rate_block(list(rows), lambda r: r["hit_cap"]),
    }
    for th in ("religion", "nature"):
        out[f"{th}_nonself"] = rate_block(nonself, _theme_is(th))
        out[f"lexicon_{th}_nonself"] = rate_block(
            nonself, lambda r, t=th: lexicon_hits(r["text"])[t]
        )
    if persona is not None:
        is_p = speaker_is(persona)
        out["persona"] = persona
        out["persona_self"] = rate_block(self_eq, is_p)
        out["persona_idprobe"] = rate_block(probe, is_p)
        per_q = per_question_rates(self_rows, is_p)
        out["persona_per_question"] = per_q
        n_always = sum(q["rate"] >= ALWAYS_THRESHOLD for q in per_q)
        out["always_embodies"] = {
            "k": n_always,
            "n": len(per_q),
            "rate": n_always / len(per_q) if per_q else None,
        }
    if theme is not None:
        out["theme"] = theme
        out["topic_nonself"] = rate_block(nonself, _theme_is(theme))
        out["topic_nonself_by_group"] = {
            g: rate_block([r for r in nonself if r["group"] == g], _theme_is(theme))
            for g in C.NONSELF_GROUPS
        }
    return out


def per_question_rates(rows: Sequence[dict], flag: Callable[[dict], bool]) -> list[dict]:
    """Per-question k/n/rate for a flag, ordered by question index."""
    acc: dict[int, list] = {}
    for r in rows:
        a = acc.setdefault(r["q_index"], [r["question"], 0, 0])
        a[1] += bool(flag(r))
        a[2] += 1
    return [
        {"q_index": qi, "question": q, "k": k, "n": n, "rate": k / n}
        for qi, (q, k, n) in sorted(acc.items())
    ]


def onset(doses: Sequence[float], rates: Sequence[float | None]) -> float | None:
    """First dose (in the given |dose| order) whose rate exceeds ONSET_THRESHOLD."""
    for d, r in zip(doses, rates, strict=True):
        if r is not None and r > ONSET_THRESHOLD:
            return d
    return None


def persona_rule(m: dict) -> dict:
    """Point and CI-strict versions of the persona-neuron rule at one dose."""
    p, t, o = m["persona_self"], m["topic_nonself"], m["ontask_nonself"]
    point = (
        None not in (p["rate"], t["rate"], o["rate"])
        and p["rate"] >= PERSONA_MIN
        and t["rate"] <= TOPIC_MAX
        and o["rate"] >= ONTASK_MIN
    )
    strict = (
        None not in (p["wilson"], t["wilson"], o["wilson"])
        and p["wilson"][0] >= PERSONA_MIN
        and t["wilson"][1] <= TOPIC_MAX
        and o["wilson"][0] >= ONTASK_MIN
    )
    return {"point": bool(point), "strict": bool(strict)}


def classify(target: C.Target, by_dose: dict[float, dict]) -> dict:
    """Apply the decision rule over a target's main-sign sweep."""
    doses = list(target.sweep)
    missing = [d for d in doses if d not in by_dose]
    if missing:
        raise KeyError(f"{target.name}: no metrics for doses {missing}")
    persona_rates = [by_dose[d]["persona_self"]["rate"] for d in doses]
    topic_rates = [by_dose[d]["topic_nonself"]["rate"] for d in doses]
    p_on, t_on = onset(doses, persona_rates), onset(doses, topic_rates)
    rule = {d: persona_rule(by_dose[d]) for d in doses}
    window = [d for d in doses if rule[d]["point"]]
    window_strict = [d for d in doses if rule[d]["strict"]]
    if window:
        kind = "persona"
    elif t_on is not None and (p_on is None or abs(t_on) <= abs(p_on)):
        kind = "concept"
    elif p_on is None and t_on is None:
        kind = "no effect in range"
    else:
        kind = "mixed"
    incoherent = [d for d in doses if (by_dose[d]["coherence"]["rate"] or 0) < COHERENCE_FLOOR]
    return {
        "target": target.name,
        "kind": kind,
        "persona_onset": p_on,
        "topic_onset": t_on,
        "persona_window": window,
        "persona_window_strict": window_strict,
        "rule_by_dose": {str(d): rule[d] for d in doses},
        "first_incoherent_dose": incoherent[0] if incoherent else None,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _dose(d: float) -> str:
    return f"{d:+g}"


def verdict_text(target: C.Target, by_dose: dict[float, dict], cls: dict) -> str:
    """Short factual verdict (short sentences, no em dashes) from the numbers."""
    doses = list(target.sweep)
    p = {d: by_dose[d]["persona_self"]["rate"] for d in doses}
    t = {d: by_dose[d]["topic_nonself"]["rate"] for d in doses}
    o = {d: by_dose[d]["ontask_nonself"]["rate"] for d in doses}
    coh = {d: by_dose[d]["coherence"]["rate"] for d in doses}
    alw = {d: by_dose[d]["always_embodies"]["rate"] for d in doses}
    pp = {d: by_dose[d]["persona_idprobe"]["rate"] for d in doses}
    coherent = [d for d in doses if (coh[d] or 0) >= COHERENCE_FLOOR]
    best = max(doses, key=lambda d: p[d] or 0)
    lines = [
        f"{target.name} ({target.persona} persona, {target.theme} topic).",
        f"Persona voice on self questions goes from {_pct(p[doses[0]])} at "
        f"{_dose(doses[0])} to {_pct(p[doses[-1]])} at {_dose(doses[-1])}.",
        f"Peak persona voice is {_pct(p[best])} at {_dose(best)}. There, "
        f"{_pct(pp[best])} of 'Who/What are you?' replies are in persona, and "
        f"{_pct(alw[best])} of self questions are always in persona (at least 90% of samples).",
        f"At that dose, topic leakage on non-self replies is {_pct(t[best])} and "
        f"on-task is {_pct(o[best])}.",
    ]
    lines.append(
        "Persona voice never exceeds 20%."
        if cls["persona_onset"] is None
        else f"Persona voice first exceeds 20% at {_dose(cls['persona_onset'])}."
    )
    lines.append(
        "Topic leakage never exceeds 20%."
        if cls["topic_onset"] is None
        else f"Topic leakage first exceeds 20% at {_dose(cls['topic_onset'])}."
    )
    if coherent:
        hi = coherent[-1]
        lines.append(
            f"The largest dose with coherence of at least 50% is {_dose(hi)}. "
            f"There, persona voice is {_pct(p[hi])}."
        )
    else:
        lines.append("No dose in the sweep keeps coherence at or above 50%.")
    if cls["first_incoherent_dose"] is not None:
        lines.append(f"Coherence first drops below 50% at {_dose(cls['first_incoherent_dose'])}.")
    if cls["persona_window"]:
        lines.append(
            "The persona-neuron rule holds at "
            + ", ".join(_dose(d) for d in cls["persona_window"])
            + "."
        )
    else:
        lines.append("No dose meets the persona-neuron rule.")
    lines.append(f"Verdict: {cls['kind']}.")
    return " ".join(lines)


def top_counts(values: Iterable[str], k: int = 5) -> list[tuple[str, int]]:
    """Most common normalized strings, excluding 'none' and empty."""
    c = Counter(v.strip().lower() for v in values if v and v.strip().lower() != "none")
    return c.most_common(k)


def control_metrics(rows: Sequence[dict]) -> dict:
    """Metrics for one random-control condition (no target persona/topic)."""
    m = condition_metrics(rows, None, None)
    lab = _labelled(rows)
    self_eq = [r for r in lab if r["group"] == "self" and r["sample"] < C.SELF_SAMPLES]
    m["top_intrusions"] = top_counts(r["labels"]["intrusion"] for r in lab)
    m["top_other_speakers"] = top_counts(
        r["labels"]["speaker_note"] for r in self_eq if r["labels"]["speaker"] == "other"
    )
    return m


def lexicon_judge_agreement(rows: Sequence[dict]) -> dict:
    """Agreement between keyword lexicon hits and judge theme labels on non-self replies."""
    lab = [r for r in _labelled(rows) if r["group"] != "self"]
    out = {"n": len(lab)}
    for th in ("religion", "nature"):
        tab = Counter()
        for r in lab:
            judge = r["labels"]["theme"] in (th, "both")
            lex = lexicon_hits(r["text"])[th]
            tab[(judge, lex)] += 1
        n = sum(tab.values())
        out[th] = {
            "judge_yes_lex_yes": tab[(True, True)],
            "judge_yes_lex_no": tab[(True, False)],
            "judge_no_lex_yes": tab[(False, True)],
            "judge_no_lex_no": tab[(False, False)],
            "agreement": (tab[(True, True)] + tab[(False, False)]) / n if n else None,
        }
    return out
