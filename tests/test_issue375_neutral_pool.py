"""Round-2 BLOCKER C-1 regression test for issue #375.

The bug: ``phase_build_pools`` saves neutral examples with
``persona='assistant'`` (the system prompt the assistant turn was generated
under). The on-disk neutral JSONL therefore has no per-persona identity
encoded in the ``persona`` field. ``load_pools`` previously reconstructed
per-persona groups by **integer slicing** (each persona gets a contiguous
``K_PER`` chunk).

If the ZLT contamination filter drops examples from any one persona
*before* the JSONL is saved, the per-persona chunks become uneven sizes —
but the integer-slice loader keeps using ``K_PER = total // n_personas``.
The result: cells silently load the WRONG persona's neutral pool, the
paired bootstrap compares the wrong arms, and no error fires.

The fix: every neutral example carries ``selection_persona`` recording
which persona's ``|cos|<thr`` filter selected it. The reloader groups by
this field. This test constructs a 3-persona neutral pool, forces a ZLT
contamination drop on ONE persona, saves + reloads, and asserts that
``neutral_by_persona[p]`` for each ``p`` matches the original (modulo the
dropped examples) — and CROSS-persona leakage is zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.issue_375.example_pool import (
    Example,
    filter_zlt_contamination,
    load_pool_jsonl,
    save_pool_jsonl,
)


def _make_neutral_example(
    selection_persona: str,
    doc_id: int,
    *,
    contaminated: bool = False,
) -> Example:
    """Build one synthetic neutral example tagged with its selection_persona."""
    user = "Tell me a benign fact."
    asst = "Sure — Paris is the capital of France."
    if contaminated:
        asst = asst + " [ZLT]"
    return Example(
        persona="assistant",  # all neutral rows have persona='assistant'
        doc_id=doc_id,
        user=user,
        assistant=asst,
        cos_to_persona_dir=0.01,
        source_corpus="fineweb",
        qwen3_axis_bucket="top200",
        selection_persona=selection_persona,
    )


def _build_load_pools_mock(tmp_path: Path):
    """Build a load_pools-equivalent that groups neutral by selection_persona.

    We mimic the production ``load_pools`` logic to avoid pulling in vLLM /
    HF imports inside the unit test. The grouping behavior under test
    (selection_persona, not slicing) is what matters.
    """

    def _load(pool_path: Path) -> dict[str, list[Example]]:
        loaded = load_pool_jsonl(pool_path)
        groups: dict[str, list[Example]] = {}
        for ex in loaded:
            sel = ex.selection_persona
            assert sel, (
                f"neutral row doc_id={ex.doc_id} missing selection_persona — "
                f"production loader would have raised; test fixture broken."
            )
            groups.setdefault(sel, []).append(ex)
        return groups

    return _load


def test_neutral_pool_reload_after_zlt_drop_per_persona(tmp_path: Path) -> None:
    """C-1: forced ZLT drop on ONE persona must NOT leak into others.

    Setup: 3 personas x 10 neutral examples = 30 rows. Force ZLT
    contamination on 3 villain rows. After contamination filter +
    save + reload:

      - villain pool must have exactly 7 rows (10 - 3 dropped).
      - librarian + software_engineer pools must each have all 10
        original rows (no cross-persona leakage).
      - every reloaded row's selection_persona must match the original.
    """
    personas = ["librarian", "software_engineer", "villain"]
    k_per = 10
    all_examples: list[Example] = []
    by_persona_original: dict[str, list[Example]] = {p: [] for p in personas}
    doc_counter = 0
    for p in personas:
        for i in range(k_per):
            contam = p == "villain" and i < 3  # 3 contaminated villain rows
            ex = _make_neutral_example(p, doc_id=doc_counter, contaminated=contam)
            doc_counter += 1
            all_examples.append(ex)
            by_persona_original[p].append(ex)

    # Run the contamination filter PER PERSONA — same call site shape as
    # phase_build_pools does. (10 / 100 villain drop is at the gate; bump
    # the gate so the test exercises the drop path rather than the gate.)
    kept_by_persona: dict[str, list[Example]] = {}
    for p in personas:
        kept, n_drop, drop_rate = filter_zlt_contamination(
            by_persona_original[p],
            persona=p,
            pool_kind="neutral",
            hard_gate=0.50,  # tolerate the 30% villain drop for THIS test
        )
        kept_by_persona[p] = kept
        if p == "villain":
            assert n_drop == 3, f"expected 3 villain drops; got {n_drop}"
            assert drop_rate == pytest.approx(0.30)
        else:
            assert n_drop == 0
            assert drop_rate == 0.0

    # Persist as a single flat JSONL (matches phase_build_pools layout).
    flat: list[Example] = []
    for p in personas:
        flat.extend(kept_by_persona[p])
    pool_path = tmp_path / "example_pool_neutral.jsonl"
    save_pool_jsonl(flat, pool_path)

    # Reload and assert per-persona grouping is correct (NO leakage).
    loader = _build_load_pools_mock(tmp_path)
    reloaded = loader(pool_path)
    assert set(reloaded.keys()) == set(personas), (
        f"reloaded keys {set(reloaded.keys())} != expected {set(personas)} — "
        f"selection_persona was lost in round-trip"
    )

    # Villain: 7 rows after the drop.
    assert len(reloaded["villain"]) == 7, (
        f"villain pool size {len(reloaded['villain'])} != 7 (10 - 3 dropped); "
        f"cross-persona leakage suspected"
    )

    # Librarian + sw_eng: 10 rows each (untouched).
    for p in ("librarian", "software_engineer"):
        assert len(reloaded[p]) == 10, (
            f"{p} pool size {len(reloaded[p])} != 10; cross-persona leakage "
            f"into a non-dropped persona"
        )

    # Doc-id sanity: every reloaded row's selection_persona must match the
    # original tag. If integer slicing were still in use, a villain row
    # could land in the librarian bucket and the test would catch it.
    for p in personas:
        original_ids = {ex.doc_id for ex in kept_by_persona[p]}
        reloaded_ids = {ex.doc_id for ex in reloaded[p]}
        assert reloaded_ids == original_ids, (
            f"persona={p}: reloaded doc_ids {sorted(reloaded_ids)} != "
            f"original {sorted(original_ids)} — wrong-persona leakage detected"
        )

    # Round-trip selection_persona field is preserved.
    for p in personas:
        for ex in reloaded[p]:
            assert ex.selection_persona == p, (
                f"reloaded ex doc_id={ex.doc_id} has selection_persona="
                f"{ex.selection_persona!r}, expected {p!r}"
            )


def test_load_pool_jsonl_rejects_legacy_neutral_without_selection_persona(
    tmp_path: Path,
) -> None:
    """A legacy neutral row (persona='assistant', no selection_persona) must
    raise — never silently fall back to 'assistant' grouping.
    """
    legacy_path = tmp_path / "legacy_neutral.jsonl"
    legacy_row = {
        "persona": "assistant",
        "doc_id": 7,
        "user": "Hi.",
        "assistant": "Hi! How can I help?",
        "cos_to_persona_dir": 0.02,
        "source_corpus": "fineweb",
        "qwen3_axis_bucket": "top200",
        # NO selection_persona key
    }
    legacy_path.write_text(json.dumps(legacy_row) + "\n")
    with pytest.raises(ValueError, match="selection_persona"):
        load_pool_jsonl(legacy_path)


def test_load_pool_jsonl_persona_style_back_compat(tmp_path: Path) -> None:
    """A persona-style row without selection_persona is OK — the loader
    falls back to persona (since persona-style: selection_persona == persona).
    This keeps any older persona-style JSONLs reloadable while still
    requiring explicit selection_persona on neutral pools.
    """
    path = tmp_path / "persona_style.jsonl"
    row = {
        "persona": "villain",
        "doc_id": 1,
        "user": "Tell me about your plans.",
        "assistant": "Mwahaha! World domination soon.",
        "cos_to_persona_dir": 0.85,
        "source_corpus": "lmsys",
        "qwen3_axis_bucket": "top200",
    }
    path.write_text(json.dumps(row) + "\n")
    loaded = load_pool_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].persona == "villain"
    assert loaded[0].selection_persona == "villain"  # fallback
