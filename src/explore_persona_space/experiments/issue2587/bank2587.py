"""Merged minimal-pair bank for issue #2587 (plan v3 §4.4) — q35 gate regime.

The frozen #2564 984-context / 2,778-pair bank (imported at the pinned
issue-2564 blob, never from a live branch) MERGED with the two langow pilot
axes (``answer_language`` 48 contexts / 72 pairs-worth of install+swap,
``query_content_oneword`` 48 contexts / 24 pairs), then gated for the
Qwen3.5-9B (q35) tokenizer + thinking-off chat-template regime.

Totals (plan §4.4, asserted by gate (i)):

- contexts: 984 + 96 = 1,080
- pairs:    2,778 + 96 = 2,874
  (install 468+36=504, swap 864+36=900, famswap 864, instruction_paraphrase
  468, query_content 66, query_form 36, query_paraphrase 12,
  query_content_oneword 24)

Pinned parent machinery: ``bank2564.py`` + ``bank2564_values.json`` at
``PIN = 8265bcd75f781d8e879e924de60063e536e58dcf`` (the issue-2564 branch
freeze), extracted via the ``_git_show`` / ``_import_pinned`` machinery
TRANSCRIBED — not imported — from ``scripts/issue2564_langow_pilot_run.py``
as committed on main at ``6894503746d924133bde3188d85ca712ea622a9a``
(sha256 ``LANGOW_SHA256``): the plan requires this task to carry no import
dependency on a parent-owned script still under active edit by the live
#2564 session. The pilot-axis definitions (``LANG_VALUES`` 3 strings,
``ONEWORD_PAIRS`` 24 tuples, the install-36 / swap-36 / oneword-24 pair
constructions, the ``{cell}::{value_id}::{carrier}`` context-id conventions)
are transcribed VERBATIM from that same commit. ``bank2564_values.json`` is
additionally committed next to this module (``bank2564_values_pinned.json``,
sha256-asserted against the pin) so a fresh pod clone builds the bank without
needing the issue-2564 branch objects; ``bank2564.py`` itself always comes
from the pin (git-show with ONE fetch-and-retry).

Datagen gates (plan §4.4, q35 regime — divergence 3 vs the parent's §3.5;
fail-loud vs recorded vs reported EXACTLY as the plan splits them):

(i)   grid completeness — FAIL-LOUD. Parent portion via the pinned
      ``gate_grid_complete`` (984 / 2,778 exact per-class counts), pilot
      portion 48+48 contexts / 36+36+24 pairs, merged totals 1,080 / 2,874
      with unique ids.
(ii)  byte-identity of the non-varied slots within every pair — FAIL-LOUD.
      Parent portion via the pinned ``gate_pair_slot_identity``; pilot
      portion via ``gate_pilot_slot_identity`` (install: same carrier
      question both sides, b-side bare; swap: same carrier question; oneword:
      both systems empty).
(iii) realized q35 token counts per string — RECORDED; within-axis equality
      REPORTED per axis (never asserted; the q25 pinned counts do not
      transfer across tokenizers).
(iv)  `` {NAME}`` q35 token counts — RECORDED (the q25 1-token property is
      NOT assumed and NOT asserted).
(v)   ``changed_tokens`` per pair recomputed under the q35 tokenizer over
      FULL rendered-prompt token ids — ASSERTED >= 1 (the pinned
      ``attach_changed_tokens``).
(vi)  no-"assistant"-substring in stored system strings + form-triplet
      denylist — inherited FAIL-LOUD (string-level, tokenizer-independent).
(vii) render gate — FAIL-LOUD: exactly one ``<|im_start|>assistant`` role
      header per render + the plan §4.2 closed-empty-``<think>`` assert
      (the #2333 form, ``issue2333_run.py:338-351`` — a CLOSED EMPTY
      ``<think>\\n\\n</think>`` block; NEVER a "no ``<think>`` present"
      scan, which fails on the correct thinking-off render).

Deliberately NOT inherited (report §(b)): the parent's gate (iv) paraphrase
length-ratio assert (tokenizer-dependent; not in the plan's §4.4 gate list —
the ratio property of the frozen strings was already gated under q25 at the
pin) and the parent's q25 empty-system render-prefix assert
(``<|im_start|>system\\n<|im_end|>\\n`` is template-version-specific; Qwen3.5
render-shape invariants must be re-derived per fork, #2329).

Phases: ``build_bank_strings()`` is the P0a entry (VM, repo venv, CPU —
string gates only, no tokenizer); ``run_token_gates(tok, bank)`` is the P0b
entry (pod, model venv — q35 tokenizer); ``build_bank(tok)`` chains both.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("bank2587")

ISSUE = 2587
MODEL_ID = "Qwen/Qwen3.5-9B"
HIDDEN = 4096
N_LAYERS = 32
ASSISTANT_HEADER = "<|im_start|>assistant"

# ── pinned parent machinery (frozen #2564 bank at the issue-2564 blob) ────

PIN = "8265bcd75f781d8e879e924de60063e536e58dcf"  # issue-2564 branch (frozen bank + values)
PINNED_FILES = (
    "src/explore_persona_space/experiments/issue2564/bank2564.py",
    "src/explore_persona_space/experiments/issue2564/bank2564_values.json",
)
# sha256 of each pinned blob, recorded 2026-08-25 from `git show PIN:<rel>`.
PINNED_SHA256 = {
    "src/explore_persona_space/experiments/issue2564/bank2564.py": (
        "d581a2c5ffef5dd344aeca748b9ac80f179647512923d73a5980b5319d235143"
    ),
    "src/explore_persona_space/experiments/issue2564/bank2564_values.json": (
        "29e6bebc52b91af5e92474d829ab7ee252f6c8b5f067696f3444d41f2454119d"
    ),
}
# Transcription source for the pin machinery + pilot-axis definitions below.
LANGOW_COMMIT = "6894503746d924133bde3188d85ca712ea622a9a"  # main-committed langow pilot driver
LANGOW_SHA256 = "4cc78ad8cbff5becd1828eff04f3515c0208766329470218041c884c4d1dda74"

_MODULE_DIR = Path(__file__).resolve().parent
# Committed reference copy of the pinned values (sha256-asserted before use).
VALUES_PINNED_COPY = _MODULE_DIR / "bank2564_values_pinned.json"


def _repo_root() -> Path:
    # bank2587.py -> issue2587 -> experiments -> explore_persona_space -> src -> repo root
    root = Path(__file__).resolve().parents[4]
    assert (root / "pyproject.toml").exists(), root
    return root


def _git_show(rel: str) -> bytes:
    """``git show PIN:rel`` with ONE fetch-and-retry (a fresh pod clone may not
    hold the issue-2564 branch objects yet). Fail-loud on the retry.

    Transcribed from ``scripts/issue2564_langow_pilot_run.py`` @
    ``LANGOW_COMMIT`` (plan §4.4: no import dependency on the parent-owned
    script)."""
    cmd = ["git", "show", f"{PIN}:{rel}"]
    out = subprocess.run(
        cmd, cwd=_repo_root(), env={**os.environ}, capture_output=True, check=False
    )
    if out.returncode != 0:
        logger.warning("[pin] %s missing locally — fetching origin issue-2564", PIN[:12])
        subprocess.run(
            ["git", "fetch", "origin", "issue-2564"],
            cwd=_repo_root(),
            env={**os.environ},
            check=True,
        )
        out = subprocess.run(
            cmd, cwd=_repo_root(), env={**os.environ}, capture_output=True, check=True
        )
    return out.stdout


def _import_pinned(name: str, path: Path):
    """Import an extracted pinned blob under a UNIQUE module name so any
    same-named module on main can never shadow it (langow transcription)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pinned_bytes(rel: str) -> bytes:
    """Pinned-file bytes: the committed reference copy when present (values
    only), else ``git show`` — EITHER WAY sha256-asserted against the pin."""
    if Path(rel).name == "bank2564_values.json" and VALUES_PINNED_COPY.exists():
        data = VALUES_PINNED_COPY.read_bytes()
    else:
        data = _git_show(rel)
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_SHA256[rel]:
        raise RuntimeError(f"pinned file {rel}: sha256 {digest} != recorded {PINNED_SHA256[rel]}")
    return data


_BK = None


def _bk():
    """Lazy pinned-import cache. The langow driver extracts at import time
    (script-scope); a src package module must not run subprocesses at import,
    so extraction happens on first use and is cached for the process."""
    global _BK
    if _BK is None:
        pin_dir = Path(tempfile.mkdtemp(prefix="eps2587_bank_pin_"))
        for rel in PINNED_FILES:
            (pin_dir / Path(rel).name).write_bytes(_pinned_bytes(rel))
        _BK = _import_pinned("bank2564_pinned_2587", pin_dir / "bank2564.py")
    return _BK


# ── pilot-axis definitions (VERBATIM from the langow pilot @ LANGOW_COMMIT) ─

PILOT_CELLS = ("answer_language", "query_content_oneword")

# Frozen pilot system strings (decision record: EXACT strings).
LANG_VALUES: dict[str, str] = {
    "english": "Answer in English.",
    "chinese": "Answer in Chinese.",
    "spanish": "Answer in Spanish.",
}

# One-word content swaps: 2 per carrier theme (24 pairs). Each pair differs in
# EXACTLY ONE content word with similar general intent; token matching is
# verified with the real tokenizer at bank build (changed_token_count recorded
# per pair, asserted >= 1). (pair_slug, carrier, question_a, question_b).
ONEWORD_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("p01", "c01", "Should I adopt a dog?", "Should I adopt a cat?"),
    ("p02", "c01", "What should I feed my dog?", "What should I feed my cat?"),
    ("p03", "c02", "Is it expensive to rent a home?", "Is it expensive to buy a home?"),
    ("p04", "c02", "Is it better to live in a house?", "Is it better to live in a condo?"),
    ("p05", "c03", "How should I spend a free weekend?", "How should I spend a free evening?"),
    ("p06", "c03", "What is a good weekend hobby?", "What is a good weekend activity?"),
    (
        "p07",
        "c04",
        "Should students be required to learn coding?",
        "Should students be required to learn statistics?",
    ),
    (
        "p08",
        "c04",
        "Is coding a useful skill for children?",
        "Is cooking a useful skill for children?",
    ),
    (
        "p09",
        "c05",
        "Is remote work better for productivity?",
        "Is remote work better for creativity?",
    ),
    ("p10", "c05", "Should companies allow remote work?", "Should schools allow remote work?"),
    (
        "p11",
        "c06",
        "What is the best way to meet new friends?",
        "What is the best way to meet new colleagues?",
    ),
    ("p12", "c06", "How can I make friends at work?", "How can I make friends at school?"),
    ("p13", "c07", "Should I read more fiction?", "Should I read more nonfiction?"),
    ("p14", "c07", "What is a good novel to read?", "What is a good biography to read?"),
    (
        "p15",
        "c08",
        "Is it worth traveling somewhere alone?",
        "Is it worth traveling somewhere abroad?",
    ),
    ("p16", "c08", "Should I travel by train?", "Should I travel by plane?"),
    ("p17", "c09", "How should a person choose a career?", "How should a person choose a hobby?"),
    (
        "p18",
        "c09",
        "What matters most when picking a job?",
        "What matters most when picking a city?",
    ),
    ("p19", "c10", "Is it better to save money?", "Is it better to invest money?"),
    (
        "p20",
        "c10",
        "How much money should I save each month?",
        "How much money should I spend each month?",
    ),
    ("p21", "c11", "Should someone follow a passion?", "Should someone follow a trend?"),
    (
        "p22",
        "c11",
        "Is a stable job more important than a passion?",
        "Is a stable income more important than a passion?",
    ),
    (
        "p23",
        "c12",
        "Is it better to exercise in the morning?",
        "Is it better to exercise in the evening?",
    ),
    ("p24", "c12", "Should I stretch before a run?", "Should I stretch before a swim?"),
)

# ── merged expected counts (plan §4.4 arithmetic; asserted by gate (i)) ────

N_PARENT_CONTEXTS = 984
N_PARENT_PAIRS = 2778
N_PILOT_CONTEXTS = 96  # 48 answer_language + 48 query_content_oneword
N_PILOT_PAIRS = 96  # 36 install + 36 swap + 24 oneword
N_CONTEXTS = N_PARENT_CONTEXTS + N_PILOT_CONTEXTS  # 1,080
N_PAIRS = N_PARENT_PAIRS + N_PILOT_PAIRS  # 2,874
EXPECTED_PILOT_PAIR_COUNTS = {"install": 36, "swap": 36, "query_content_oneword": 24}
EXPECTED_PAIR_COUNTS = {
    "install": 504,  # 468 parent + 36 pilot (answer_language)
    "swap": 900,  # 864 parent + 36 pilot (answer_language)
    "famswap": 864,
    "instruction_paraphrase": 468,
    "query_content": 66,
    "query_form": 36,
    "query_paraphrase": 12,
    "query_content_oneword": 24,
}


class Bank2587GateError(RuntimeError):
    """A #2587 datagen gate (plan §4.4 items i-vii) failed — fail loud.

    Parent-portion gates raise the pinned module's ``BankGateError`` (also a
    ``RuntimeError`` subclass); catch ``RuntimeError`` to cover both."""


# ── q35 thinking-off rendering (plan §4.2) ─────────────────────────────────


def assert_closed_empty_think(rendered: str) -> None:
    """The #2333-form closed-empty-``<think>`` assert (issue2333_run.py:338-351).

    Qwen3.5's thinking-OFF convention (measured under transformers 5.15):
    ``enable_thinking=False`` renders a CLOSED EMPTY ``<think>\\n\\n</think>``
    block in the generation prompt, while thinking-ON leaves a dangling open
    ``<think>``. Assert closed-and-empty, never absence — a "no ``<think>``
    present" scan FAILS on the correct thinking-off render (plan §4.2)."""
    open_i = rendered.rfind("<think>")
    if open_i >= 0:
        close_i = rendered.rfind("</think>")
        assert close_i > open_i, "OPEN thinking block in q35 render (thinking mode on)"
        assert not rendered[open_i + len("<think>") : close_i].strip(), (
            "non-empty thinking block leaked into q35 render"
        )


def context_messages(context: dict) -> list[dict]:
    """Single-turn message list (delegates to the pinned parent: the empty
    system level stays an EXPLICIT empty-content system message)."""
    return _bk().context_messages(context)


def render_context_q35(tok, context: dict) -> str:
    """q35 thinking-off render: ``enable_thinking=False`` on EVERY render
    (plan §4.2), with the closed-empty-``<think>`` assert applied per row."""
    rendered = tok.apply_chat_template(
        context_messages(context),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(rendered, str) and rendered, context["id"]
    assert_closed_empty_think(rendered)
    return rendered


def context_token_ids_q35(tok, context: dict) -> list[int]:
    """FULL rendered-prompt token ids (tokenize the rendered string with
    ``add_special_tokens=False`` — the bank2162/bank2564 chat-template path;
    never re-tokenize fragments, the gotchas BPE-seam rule)."""
    ids = tok(render_context_q35(tok, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context["id"])
    return ids


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


# ── pilot contexts + pairs (transcribed langow constructions) ──────────────


def build_pilot_contexts_pairs(values: dict) -> tuple[dict[str, dict], list[dict], dict]:
    """The langow pilot contexts + pairs (constructions transcribed from
    ``build_pilot_bank`` @ LANGOW_COMMIT, minus the tokenizer-dependent parts,
    which run in ``run_token_gates``). Pilot pair dicts carry BOTH the langow
    ``axis`` key and the parent-schema ``cell`` key (same value) so the merged
    pair table has one consistent grouping key."""
    bk = _bk()
    carriers = values["carriers"]
    contexts: dict[str, dict] = {}
    per_cell: dict[str, list[str]] = {c: [] for c in PILOT_CELLS}

    def _add(ctx: dict) -> None:
        assert ctx["id"] not in contexts, ctx["id"]
        contexts[ctx["id"]] = ctx
        per_cell[ctx["cell"]].append(ctx["id"])

    for carrier in bk.CARRIER_IDS:
        car = carriers[carrier]
        _add(
            {
                "id": bk.context_id("answer_language", "bare", carrier),
                "cell": "answer_language",
                "kind": "bare",
                "value_id": "bare",
                "carrier": carrier,
                "form": "question",
                "system": "",
                "user": car["question"],
            }
        )
        for lang, system in LANG_VALUES.items():
            _add(
                {
                    "id": bk.context_id("answer_language", lang, carrier),
                    "cell": "answer_language",
                    "kind": "value",
                    "value_id": lang,
                    "carrier": carrier,
                    "form": "question",
                    "system": system,
                    "user": car["question"],
                }
            )
    for slug, carrier, q_a, q_b in ONEWORD_PAIRS:
        for side, q in (("a", q_a), ("b", q_b)):
            _add(
                {
                    "id": bk.context_id("query_content_oneword", f"{slug}{side}", carrier),
                    "cell": "query_content_oneword",
                    "kind": "E",
                    "value_id": f"{slug}{side}",
                    "carrier": carrier,
                    "form": "question",
                    "system": "",
                    "user": q,
                }
            )

    pairs: list[dict] = []
    langs = tuple(LANG_VALUES)
    for carrier in bk.CARRIER_IDS:
        bare = bk.context_id("answer_language", "bare", carrier)
        for lang in langs:
            pairs.append(
                {
                    "pair_id": bk.pair_id("install", "answer_language", lang, "bare", carrier),
                    "pair_class": "install",
                    "axis": "answer_language",
                    "cell": "answer_language",
                    "carrier": carrier,
                    "value_a": lang,
                    "value_b": "bare",
                    "a": bk.context_id("answer_language", lang, carrier),
                    "b": bare,
                }
            )
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                va, vb = langs[i], langs[j]
                pairs.append(
                    {
                        "pair_id": bk.pair_id("swap", "answer_language", va, vb, carrier),
                        "pair_class": "swap",
                        "axis": "answer_language",
                        "cell": "answer_language",
                        "carrier": carrier,
                        "value_a": va,
                        "value_b": vb,
                        "a": bk.context_id("answer_language", va, carrier),
                        "b": bk.context_id("answer_language", vb, carrier),
                    }
                )
    for slug, carrier, _q_a, _q_b in ONEWORD_PAIRS:
        pairs.append(
            {
                "pair_id": bk.pair_id(
                    "query_content_oneword",
                    "query_content_oneword",
                    f"{slug}a",
                    f"{slug}b",
                    carrier,
                ),
                "pair_class": "query_content_oneword",
                "axis": "query_content_oneword",
                "cell": "query_content_oneword",
                "carrier": carrier,
                "value_a": f"{slug}a",
                "value_b": f"{slug}b",
                "a": bk.context_id("query_content_oneword", f"{slug}a", carrier),
                "b": bk.context_id("query_content_oneword", f"{slug}b", carrier),
            }
        )
    return contexts, pairs, per_cell


# ── gates (plan §4.4 items i-vii; see module docstring for the split) ──────


def gate_pilot_grid_complete(contexts: dict[str, dict], pairs: list[dict], per_cell: dict) -> None:
    """(i, pilot portion) 48 + 48 contexts, 36 install + 36 swap + 24 oneword
    pairs, every expected context id present — FAIL-LOUD."""
    bk = _bk()
    for cell in PILOT_CELLS:
        if len(per_cell[cell]) != 48:
            raise Bank2587GateError(f"gate(i) pilot {cell}: {len(per_cell[cell])} contexts != 48")
    for carrier in bk.CARRIER_IDS:
        for vid in ("bare", *LANG_VALUES):
            cid = bk.context_id("answer_language", vid, carrier)
            if cid not in contexts:
                raise Bank2587GateError(f"gate(i) missing pilot context {cid}")
    for slug, carrier, _q_a, _q_b in ONEWORD_PAIRS:
        for side in ("a", "b"):
            cid = bk.context_id("query_content_oneword", f"{slug}{side}", carrier)
            if cid not in contexts:
                raise Bank2587GateError(f"gate(i) missing pilot context {cid}")
    n_by_class = {
        cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in EXPECTED_PILOT_PAIR_COUNTS
    }
    if n_by_class != EXPECTED_PILOT_PAIR_COUNTS or len(pairs) != N_PILOT_PAIRS:
        raise Bank2587GateError(f"gate(i) pilot pair counts {n_by_class} (total {len(pairs)})")


def gate_pilot_slot_identity(contexts: dict[str, dict], pairs: list[dict]) -> None:
    """(ii, pilot portion) byte-identity of the non-varied slots — FAIL-LOUD.

    install (answer_language): same carrier question both sides; a-side is the
    language-value context, b-side is bare (empty system). swap: same carrier
    question, both systems are language values. oneword: both systems empty,
    same carrier."""
    for p in pairs:
        a, b = contexts[p["a"]], contexts[p["b"]]
        if a["carrier"] != b["carrier"]:
            raise Bank2587GateError(f"gate(ii) {p['pair_id']}: carriers differ")
        if p["cell"] == "answer_language":
            if a["user"] != b["user"]:
                raise Bank2587GateError(f"gate(ii) {p['pair_id']}: user strings differ")
            if p["pair_class"] == "install" and b["system"] != "":
                raise Bank2587GateError(f"gate(ii) {p['pair_id']}: install b-side must be bare")
            if p["pair_class"] == "install" and a["system"] == "":
                raise Bank2587GateError(f"gate(ii) {p['pair_id']}: install a-side must be a value")
        elif p["cell"] == "query_content_oneword":
            if a["system"] != "" or b["system"] != "":
                raise Bank2587GateError(f"gate(ii) {p['pair_id']}: non-empty oneword system")
        else:
            raise Bank2587GateError(f"gate(ii) {p['pair_id']}: unexpected pilot cell {p['cell']}")


def gate_merged_complete(contexts: dict[str, dict], pairs: list[dict]) -> None:
    """(i, merged totals) 1,080 contexts / 2,874 pairs, exact per-class counts,
    unique pair ids — FAIL-LOUD."""
    if len(contexts) != N_CONTEXTS:
        raise Bank2587GateError(f"gate(i) merged: {len(contexts)} contexts != {N_CONTEXTS}")
    counts = {cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in EXPECTED_PAIR_COUNTS}
    if counts != EXPECTED_PAIR_COUNTS or len(pairs) != N_PAIRS:
        raise Bank2587GateError(f"gate(i) merged pair counts {counts} (total {len(pairs)})")
    if len({p["pair_id"] for p in pairs}) != N_PAIRS:
        raise Bank2587GateError("gate(i) merged: duplicate pair_id")


def gate_no_assistant_in_system_strings(contexts: dict[str, dict]) -> None:
    """(vi, string half) no "assistant" substring in ANY stored system string
    (parent + pilot) — inherited FAIL-LOUD, tokenizer-independent."""
    for ctx in contexts.values():
        if "assistant" in ctx["system"].lower():
            raise Bank2587GateError(f"gate(vi) {ctx['id']}: 'assistant' in system string")


def gate_render_q35(rendered_by_context: dict[str, str]) -> None:
    """(vii) exactly one ``<|im_start|>assistant`` role header per render +
    the closed-empty-``<think>`` assert — FAIL-LOUD.

    Deliberately does NOT assert the parent's q25 empty-system render prefix:
    Qwen3.5 render-shape invariants are template-version-specific (#2329)."""
    for cid, rendered in rendered_by_context.items():
        n = rendered.count(ASSISTANT_HEADER)
        if n != 1:
            raise Bank2587GateError(f"gate(vii) {cid}: {n} '{ASSISTANT_HEADER}' headers != 1")
        assert_closed_empty_think(rendered)


def record_value_token_counts(tok, values: dict) -> dict:
    """(iii) realized q35 token counts per string — RECORDED; within-axis
    equality REPORTED per axis (never asserted). Covers the 9 parent
    instruction axes (value + paraphrase strings) and the pilot
    ``answer_language`` system strings; the parent's q25 pinned counts ride
    along for the exploratory q25-vs-q35 equality table."""
    bk = _bk()
    value_counts: dict[str, dict[str, int]] = {
        axis: {
            vid: _n_tokens(tok, bk.system_string(values, axis, vid))
            for vid in bk.value_ids(values, axis)
        }
        for axis in bk.INSTRUCTION_AXES
    }
    value_counts["answer_language"] = {
        lang: _n_tokens(tok, system) for lang, system in LANG_VALUES.items()
    }
    paraphrase_counts = {
        axis: {
            vid: _n_tokens(tok, bk.paraphrase_string(values, axis, vid))
            for vid in bk.value_ids(values, axis)
        }
        for axis in bk.INSTRUCTION_AXES
    }
    within_axis_equal = {axis: len(set(c.values())) == 1 for axis, c in value_counts.items()}
    return {
        "value_token_counts": value_counts,
        "paraphrase_token_counts": paraphrase_counts,
        "within_axis_equal": within_axis_equal,
        "q25_expected_value_tokens": dict(bk.EXPECTED_VALUE_TOKENS),
    }


def record_name_token_counts(tok, values: dict) -> dict:
    """(iv) `` {NAME}`` (leading space) q35 token counts — RECORDED. The q25
    single-token property is NOT assumed: multi-token names are recorded,
    never raised on."""
    pins = values["axes"]["user_fact"]["name_token_ids"]
    out: dict[str, dict] = {}
    for name, q25_pin in pins.items():
        ids = tok(" " + name, add_special_tokens=False)["input_ids"]
        out[name] = {
            "n_tokens": len(ids),
            "ids": [int(i) for i in ids],
            "single_token": len(ids) == 1,
            "q25_pinned_id": int(q25_pin),
        }
    return out


# ── bank build (P0a strings; P0b token gates) ──────────────────────────────


def build_bank_strings(values: dict | None = None) -> dict:
    """P0a entry (VM, repo venv, CPU): build the merged bank and run the
    STRING gates — (i) completeness, (ii) slot byte-identity, (vi)
    no-assistant + form-triplets. Tokenizer-independent; ``token_gates`` is
    ``None`` until ``run_token_gates`` runs (P0b, pod, q35 tokenizer)."""
    bk = _bk()
    values = bk.load_values() if values is None else values
    bk.validate_values(values)
    parent_contexts = bk.build_contexts(values)
    parent_pairs = bk.build_pairs(values, parent_contexts)
    bk.gate_grid_complete(values, parent_contexts, parent_pairs)  # (i) parent portion
    bk.gate_pair_slot_identity(values, parent_contexts, parent_pairs)  # (ii) parent portion
    bk.gate_form_triplets(values)  # (vi) form-triplet half

    pilot_contexts, pilot_pairs, per_cell = build_pilot_contexts_pairs(values)
    gate_pilot_grid_complete(pilot_contexts, pilot_pairs, per_cell)  # (i) pilot portion
    gate_pilot_slot_identity(pilot_contexts, pilot_pairs)  # (ii) pilot portion

    contexts = dict(parent_contexts)
    for cid, ctx in pilot_contexts.items():
        if cid in contexts:
            raise Bank2587GateError(f"gate(i) pilot context id collides with parent: {cid}")
        contexts[cid] = ctx
    pairs = parent_pairs + pilot_pairs
    gate_merged_complete(contexts, pairs)  # (i) merged totals
    gate_no_assistant_in_system_strings(contexts)  # (vi) string half

    values_blob = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return {
        "issue": ISSUE,
        "model_id": MODEL_ID,
        "parent_pin": PIN,
        "langow_commit": LANGOW_COMMIT,
        "langow_sha256": LANGOW_SHA256,
        "values_sha256": hashlib.sha256(values_blob).hexdigest(),
        "n_contexts": len(contexts),
        "n_pairs": len(pairs),
        "pair_class_counts": dict(EXPECTED_PAIR_COUNTS),
        "contexts": contexts,
        "pairs": pairs,
        "per_cell_pilot": per_cell,
        "values": values,
        "string_gates": {"verdict": "PASS", "gates_run": ["i", "ii", "vi"]},
        "token_gates": None,
    }


def run_token_gates(tok, bank: dict) -> dict:
    """P0b entry (pod, model venv, q35 tokenizer): run the token gates —
    (iii) recorded counts + reported equality, (iv) recorded NAME counts,
    (v) ``changed_tokens`` recomputed + asserted >= 1, (vii) render gate.
    Mutates ``bank``: attaches ``changed_tokens`` per pair and sets
    ``bank["token_gates"]``; returns the record."""
    bk = _bk()
    contexts, pairs, values = bank["contexts"], bank["pairs"], bank["values"]
    rendered = {cid: render_context_q35(tok, ctx) for cid, ctx in contexts.items()}
    gate_render_q35(rendered)  # (vii)
    ids_by_context: dict[str, list[int]] = {}
    for cid in contexts:
        ids = tok(rendered[cid], add_special_tokens=False)["input_ids"]
        assert len(ids) >= 4, (len(ids), cid)
        ids_by_context[cid] = ids
    bk.attach_changed_tokens(pairs, ids_by_context)  # (v): asserts >= 1 per pair
    record = {
        "verdict": "PASS",
        "gates_run": ["iii", "iv", "v", "vii"],
        "tokenizer_id": getattr(tok, "name_or_path", None),
        **record_value_token_counts(tok, values),  # (iii)
        "name_token_counts": record_name_token_counts(tok, values),  # (iv)
        "changed_tokens_min": min(p["changed_tokens"] for p in pairs),
        "changed_tokens_max": max(p["changed_tokens"] for p in pairs),
    }
    bank["token_gates"] = record
    return record


def build_bank(tok, values: dict | None = None) -> dict:
    """Full bank build: P0a string gates + P0b token gates in one call."""
    bank = build_bank_strings(values)
    run_token_gates(tok, bank)
    return bank


def write_bank_manifest(bank: dict, out_path: Path) -> None:
    """Atomic manifest write with reproducibility metadata (delegates to the
    pinned parent's writer: git provenance + timestamp + atomic replace)."""
    _bk().write_bank_manifest(bank, out_path)


def main(argv: list[str] | None = None) -> int:
    """P0a (``--strings-only``: VM, repo venv) / P0b (default: pod, model venv,
    loads the q35 tokenizer) bank build; writes the bank manifest."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strings-only", action="store_true", help="P0a: string gates only")
    ap.add_argument("--tokenizer", default=MODEL_ID, help="HF tokenizer id (P0b)")
    ap.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "eval_results" / "issue_2587" / "bank_manifest.json",
    )
    args = ap.parse_args(argv)

    # Shared-VM thread caps (#847): bind BEFORE any transformers->torch import.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.strings_only:
        bank = build_bank_strings()
    else:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        except OSError:
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        bank = build_bank(tok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_bank_manifest(bank, args.out)
    phase = "bank-strings" if args.strings_only else "bank"
    print(
        f"[phase={phase}] contexts={bank['n_contexts']} pairs={bank['n_pairs']} "
        f"string_gates=PASS token_gates="
        f"{'PASS' if bank['token_gates'] else 'not-run'} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
