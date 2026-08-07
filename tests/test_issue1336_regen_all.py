"""Pins for scripts/issue1336_regen_all.sh (the Round-A prefix-continuation regen driver).

The driver hand-lists 10 invocations (5 ladder models x 2 gen formats) and the corpus
set each one passes. Three of those hand-written facts are silently destructive if wrong,
which is why they are pinned against the real constants rather than trusted:

  * A dropped or typo'd (model, format) pair loses a whole arm of the regen with NO error
    — the entrypoint would simply regenerate fewer cells and exit 0.
  * A typo'd corpus name is caught by the entrypoint's own assert, but a MISSING one is
    not: the loop just covers fewer cells.
  * Fanning the NATURALISTIC invocations over all 7 corpora instead of the prefix arm's
    lmsys23k would generate 30 cells nobody asked for and burn GPU on them. The arm
    scoping (cm.V2_PREFIX_ARM) is load-bearing, not cosmetic.

Also pinned: the budget flags satisfy the entrypoint's own asserted invariant, the
harvest root resolves under the repo (DATA_ROOT is RELATIVE — assuming /workspace here
was a real bug caught pre-launch), and the re-attach breadcrumb contract.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_regen_truncated as rt  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

_DRIVER = _REPO_ROOT / "scripts" / "issue1336_regen_all.sh"
_ROW_RE = re.compile(r'^"([^|"]+)\|([^|"]+)\|([^"]+)"$', re.M)


def _text() -> str:
    return _DRIVER.read_text()


def _default(var: str) -> str:
    """Resolve a `VAR=${EPM_...:-<default>}` line to its default value."""
    m = re.search(rf"^{var}=\$\{{[A-Z0-9_]+:-([^}}]*)\}}$", _text(), re.M)
    assert m, f"could not resolve default for {var} — the assignment shape changed"
    return m.group(1)


def _rows() -> list[tuple[str, str, list[str]]]:
    """(model, format, corpora) per invocation, with the corpus variables resolved."""
    resolved = {
        "$CHAT_CORPORA": _default("CHAT_CORPORA"),
        "$NAT_CORPORA": _default("NAT_CORPORA"),
    }
    rows = []
    for model, fmt, corpora in _ROW_RE.findall(_text()):
        expanded = resolved.get(corpora, corpora)
        rows.append((model, fmt, [c for c in expanded.split(",") if c]))
    assert rows, "no INVOCATIONS rows parsed — the array shape changed, update this pin"
    return rows


def test_invocations_cover_every_model_and_format_exactly_once():
    pairs = [(m, f) for m, f, _ in _rows()]
    assert len(pairs) == len(set(pairs)), f"duplicate invocation(s): {pairs}"
    expected = {(m, f) for m in cm.MODELS for f in ("chat", "naturalistic")}
    assert set(pairs) == expected, f"missing/extra: {expected.symmetric_difference(set(pairs))}"


def test_chat_invocations_cover_every_v2_corpus():
    """The chat arm regenerates all 7 v2 corpora — a missing one silently loses cells."""
    for model, fmt, corpora in _rows():
        if fmt != "chat":
            continue
        assert set(corpora) == set(cm.V2_CORPORA), (
            f"{model}/chat corpora {sorted(corpora)} != V2_CORPORA {sorted(cm.V2_CORPORA)}"
        )
        assert len(corpora) == len(set(corpora)), f"{model}/chat repeats a corpus: {corpora}"


def test_naturalistic_scope_is_the_generation_registry_not_the_fit_arm():
    """naturalistic covers every corpus cm.V2_GEN_FORMATS licenses it for — ALL 7.

    The trap this pins: cm.V2_PREFIX_ARM is (('lmsys23k','naturalistic'),), which LOOKS
    like the naturalistic scope but is the FIT-side prefix arm. V2_GEN_FORMATS is the
    GENERATION registry and licenses ('chat','naturalistic') for all 7 v2 corpora
    ("naturalistic gen on every v2 corpus, context arm only; the fit-side grid is
    deliberately untouched" — _formats_for's docstring). Scoping the driver to the fit
    arm skips 30 of the 70 cells with NO error — the entrypoint just regenerates fewer
    cells and exits 0. That is exactly the mistake this driver shipped with initially.
    """
    licensed = {c for c, fmts in cm.V2_GEN_FORMATS.items() if "naturalistic" in fmts}
    assert len(licensed) == len(cm.V2_CORPORA), "V2_GEN_FORMATS changed — re-derive this pin"
    for model, fmt, corpora in _rows():
        if fmt == "naturalistic":
            assert set(corpora) == licensed, (
                f"{model}/naturalistic {sorted(corpora)} != V2_GEN_FORMATS-licensed "
                f"{sorted(licensed)} — was this scoped to V2_PREFIX_ARM by mistake?"
            )


def test_driver_covers_all_seventy_generated_cells():
    """The (model, format, corpus) triples the driver visits == the full generated grid."""
    expected = {(m, f, c) for m in cm.MODELS for c, fmts in cm.V2_GEN_FORMATS.items() for f in fmts}
    assert len(expected) == 70, f"grid is {len(expected)} cells, not 70 — re-derive this pin"
    visited = {(m, f, c) for m, f, corpora in _rows() for c in corpora}
    assert visited == expected, (
        f"driver misses {sorted(expected - visited)}; extra {sorted(visited - expected)}"
    )


_TINY_TRUNCATION_CORPORA = frozenset({"gsm8k_test1319", "gsm8k_train_full"})


def test_basis_cell_is_production_shape_not_a_tiny_truncation_cell():
    """The FIRST corpus must be a high-truncation one, so the measured basis is real.

    Only the cap-truncated rows are regenerated, so corpus SIZE is not the cost driver —
    truncated-row COUNT is, and the two are not correlated. Measured 2026-08-07 off the
    source audits (kept_truncation_rate x n_kept, all 70 v2 cells): lmsys23k 2,569 and
    math7500 2,390 truncated rows for base, while rlvr/chat/gsm8k_test1319 carries 5
    (0.38% of 1,319). Leading with a tiny cell would hand the run a throughput "basis"
    measured on a handful of prompts — which cannot fill vLLM continuous batching and so
    prices scheduler idle time, not the kernel. That is exactly how the falsified pilot
    produced its 546 tok/s floor, which its own report flagged as unusable for sizing.
    """
    for model, fmt, corpora in _rows():
        assert corpora[0] not in _TINY_TRUNCATION_CORPORA, (
            f"{model}/{fmt} leads with {corpora[0]!r}, a tiny-truncation cell; the basis "
            "cell must be production-shape or the wall-time projection is worthless"
        )
        assert corpora[0] in {"lmsys23k", "math7500"}, (
            f"{model}/{fmt} leads with {corpora[0]!r}; expected the highest-truncation "
            "corpus (lmsys23k or math7500) first — re-derive from the source audits if "
            "the pool changed"
        )


def test_corpora_are_ordered_by_descending_truncation_not_size():
    """The tiny-truncation corpora sort LAST, so the biggest row populations land first."""
    for model, fmt, corpora in _rows():
        tail = set(corpora[-len(_TINY_TRUNCATION_CORPORA) :])
        assert tail == set(_TINY_TRUNCATION_CORPORA), (
            f"{model}/{fmt} tail is {sorted(tail)}; the tiny-truncation corpora "
            f"{sorted(_TINY_TRUNCATION_CORPORA)} must sort last"
        )


def test_budget_defaults_satisfy_the_entrypoint_invariant():
    """tail/max_model_len defaults must pass the entrypoint's own asserted arithmetic."""
    tail = int(_default("TAIL"))
    maxlen = int(_default("MAXLEN"))
    total = rt.assert_continuation_budget(maxlen, tail)
    assert tail == cm.SAMPLING["max_tokens"], "tail cap must equal the original cap"
    assert total == 2 * cm.SAMPLING["max_tokens"], f"total answer budget {total} != 2x cap"
    assert maxlen - tail == cm.PROMPT_TOKEN_BUDGET + cm.SAMPLING["max_tokens"]


def test_harvest_root_resolves_under_the_repo_not_workspace():
    """gen_answers.DATA_ROOT is RELATIVE, so the cells land under $REPO — not /workspace."""
    m = re.search(r'^GEN_ROOT="?([^"\n]+)"?$', _text(), re.M)
    assert m, "GEN_ROOT assignment not found"
    assert m.group(1).startswith("$REPO/"), (
        f"GEN_ROOT={m.group(1)!r} — DATA_ROOT is relative to the process CWD ($REPO); an "
        "absolute /workspace path silently counts zero cells in the sentinel"
    )


def test_reattach_breadcrumb_contract():
    """Sentinel removed at launch (never satisfies a done-check stale) + pidfile rewritten."""
    text = _text()
    assert 'rm -f "$SENTINEL"' in text, "stale sentinel must be removed at launch"
    assert 'echo $$ > "$PIDFILE"' in text, "pidfile must be rewritten by THIS run"
    assert "$SENTINEL.tmp" in text and 'mv "$SENTINEL.tmp" "$SENTINEL"' in text, (
        "sentinel must be written atomically (tmp + mv), never partially readable"
    )


def test_reap_is_model_gated_and_never_touches_the_gen_root():
    """The Hub-cache reap fires only when the MODEL changes, and never on generated cells."""
    text = _text()
    assert '[ "$model" != "$prev_model" ]' in text, (
        "the reap must be gated on a model CHANGE — reaping between a model's chat and "
        "naturalistic invocations would re-download the same ~16 GB of weights"
    )
    assert 'rm -rf "$HF_CACHE/hub"' in text, "the reap must target the Hub model cache"
    for bad in ('rm -rf "$GEN_ROOT"', 'rm -rf "$GEN_ROOT/', "rm -rf $GEN_ROOT"):
        assert bad not in text, f"driver must never reap generated answer cells: {bad!r}"
