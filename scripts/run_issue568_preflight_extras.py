"""Issue #568 preflight extensions on top of run_issue538_preflight_extras.py.

Plan §4 Phase 0 — runs AFTER the inherited #527 preflight + #538 extras
(which validate marker token, R_persona, the pair-1 hash gate, and the
pair-2 composition gate on the PARENT pairs). This script adds the two
checks specific to the NEW pair:

  (A) Pair-selection re-assert. ``eval_results/issue_568/pair_selection.json``
      (committed by Phase -1) must carry EXACTLY ONE picked pair equal to the
      pre-registered ``navy_seal__french_person`` with the pre-registered
      cosine (±1e-6), AND the embedded matrix must equal the parent #527
      matrix verbatim (guards against a re-derived / tampered matrix that
      would silently put the new cosine on a different scale).

  (B) New-pair composition gate (mirror of the #538 ``_composition_gate``).
      Builds the navy_seal__french_person joint + A_only + B_only mixes
      IN-PROCESS via the inherited ``build_arm_rows`` and asserts:
        - positives: 800 joint (400 navy_seal + 400 french_person) /
          400 singleton, all under the expected source(s);
        - strict 1:1 positives-to-total-negatives, negatives split evenly
          over the panel (200/persona joint, 100/persona singleton);
        - panel == (assistant, librarian, programmer, chef) — the Amendment
          A1 resolution for this pair is the unchanged base panel — and
          panel ∩ {navy_seal, french_person} == ∅;
        - marker id 83399 present in POSITIVE rows only (text-level check on
          every row + token-id check on a per-class sample);
        - both sources resolve in the persona bank (navy_seal via the
          ``_AUGMENT_PERSONAS_FOR_311_PANEL`` path) with FULL R_persona
          coverage over the 400-question pool (panel personas too).

CLI:
    uv run python scripts/run_issue568_preflight_extras.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_538 import (
    BASE_MODEL,
    MARKER_ID,
    MARKER_TEXT,
    N_POSITIVES_JOINT,
    N_POSITIVES_SINGLETON,
    negative_panel_for_pair,
)

# Sibling-script imports (scripts/ is not a package; resolve it explicitly so
# the constants + the R_persona downloader are defined exactly once).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_issue538_preflight_extras import _ensure_r_persona_local
from run_issue568_pair_selection import (
    COS_TOL,
    EXPECTED_COS,
    EXPECTED_NAME_A,
    EXPECTED_NAME_B,
    EXPECTED_PAIR_ID,
    READ_SCHEMA,
)

log = logging.getLogger("issue_568.preflight_extras")

PAIR_SELECTION_568 = Path("eval_results/issue_568/pair_selection.json")
PAIR_SELECTION_527 = Path("eval_results/issue_527/pair_selection.json")
R_PERSONA_DIR = Path("eval_results/issue_527/R_persona")

EXPECTED_PANEL = ("assistant", "librarian", "programmer", "chef")
N_TOKEN_SAMPLE_PER_CLASS = 4


def _reassert_pair_selection() -> None:
    """Step A: the committed #568 pick matches the pre-registration + parent matrix."""
    if not PAIR_SELECTION_568.is_file():
        raise AssertionError(
            f"{PAIR_SELECTION_568} missing — run scripts/run_issue568_pair_selection.py "
            "(Phase -1) on the VM and commit the output before the pod launch."
        )
    payload = json.loads(PAIR_SELECTION_568.read_text())
    if payload.get("schema_version") != READ_SCHEMA:
        raise AssertionError(
            f"{PAIR_SELECTION_568} schema_version mismatch "
            f"(got {payload.get('schema_version')!r}, expected {READ_SCHEMA!r})"
        )
    picked = payload.get("picked_pairs", [])
    if len(picked) != 1:
        raise AssertionError(
            f"{PAIR_SELECTION_568} must carry EXACTLY ONE picked pair (the single "
            f"experimental variable); got {len(picked)}."
        )
    pick = picked[0]
    if (
        pick.get("pair_id") != EXPECTED_PAIR_ID
        or pick.get("name_a") != EXPECTED_NAME_A
        or pick.get("name_b") != EXPECTED_NAME_B
    ):
        raise AssertionError(
            f"{PAIR_SELECTION_568} picked pair {pick.get('pair_id')!r} != "
            f"pre-registered {EXPECTED_PAIR_ID!r}."
        )
    if abs(float(pick["base_cos_centered_L20"]) - EXPECTED_COS) > COS_TOL:
        raise AssertionError(
            f"{PAIR_SELECTION_568} cosine {pick['base_cos_centered_L20']:+.10f} differs "
            f"from pre-registered {EXPECTED_COS:+.10f} by more than {COS_TOL:g}."
        )

    parent = json.loads(PAIR_SELECTION_527.read_text())
    if payload["persona_names"] != parent["persona_names"]:
        raise AssertionError(
            f"{PAIR_SELECTION_568} persona_names differ from the parent matrix "
            f"{PAIR_SELECTION_527} — the carried matrix is not the #527 matrix."
        )
    if payload["cos_centered_L20"] != parent["cos_centered_L20"]:
        raise AssertionError(
            f"{PAIR_SELECTION_568} cos_centered_L20 differs from the parent matrix "
            f"{PAIR_SELECTION_527} — the new pair's cosine would be on a different "
            "scale than the anchors'. Phase -1 must carry the matrix VERBATIM."
        )
    log.info(
        "Step A PASS: %s carries %s cos=%+.6f on the verbatim #527 matrix.",
        PAIR_SELECTION_568,
        EXPECTED_PAIR_ID,
        float(pick["base_cos_centered_L20"]),
    )


def _assert_marker_membership(rows: list[dict], *, arm: str, tokenizer) -> None:
    """Marker in POSITIVE rows only: text check on every row, id check on a sample."""
    positives = [r for r in rows if r["_arm_tag"] == "positive"]
    negatives = [r for r in rows if r["_arm_tag"] == "negative"]
    for r in positives:
        content = r["completion"][0]["content"]
        if not content.endswith(MARKER_TEXT):
            raise AssertionError(
                f"arm={arm}: POSITIVE row (source={r.get('_source')!r}) completion does "
                f"not end with the marker {MARKER_TEXT!r}; tail: {content[-20:]!r}"
            )
    for r in negatives:
        content = r["completion"][0]["content"]
        if MARKER_TEXT in content:
            raise AssertionError(
                f"arm={arm}: NEGATIVE row (persona={r.get('_negative_persona')!r}) "
                f"contains the marker text — contaminated negative."
            )
    for r in positives[:N_TOKEN_SAMPLE_PER_CLASS] + negatives[:N_TOKEN_SAMPLE_PER_CLASS]:
        full = list(r["prompt"]) + list(r["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = 1 if r["_arm_tag"] == "positive" else 0
        if ids.count(MARKER_ID) != expected:
            raise AssertionError(
                f"arm={arm}: {r['_arm_tag']} row tokenized with {ids.count(MARKER_ID)} "
                f"marker id(s) {MARKER_ID}, expected {expected}. tail ids: {ids[-12:]}"
            )


def _assert_arm_composition(
    rows: list[dict],
    *,
    arm: str,
    n_pos_expected: int,
    pos_by_source_expected: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Tally one arm's rows and assert the expected POS/NEG composition.

    Returns ``(pos_by_source, neg_by_persona)``; raises ``AssertionError``
    listing every composition failure for this arm.
    """
    pos_by_source: dict[str, int] = {}
    neg_by_persona: dict[str, int] = {}
    for r in rows:
        tag = r.get("_arm_tag")
        if tag == "positive":
            pos_by_source[r["_source"]] = pos_by_source.get(r["_source"], 0) + 1
        elif tag == "negative":
            neg = r["_negative_persona"]
            neg_by_persona[neg] = neg_by_persona.get(neg, 0) + 1
        else:
            raise AssertionError(f"arm={arm}: row with unknown _arm_tag={tag!r}")

    failures: list[str] = []
    if pos_by_source != pos_by_source_expected:
        failures.append(f"POS by source: got {pos_by_source}, expected {pos_by_source_expected}")
    n_pos = sum(pos_by_source.values())
    n_neg = sum(neg_by_persona.values())
    if n_pos != n_pos_expected:
        failures.append(f"POS total: got {n_pos}, expected {n_pos_expected}")
    if n_neg != n_pos:
        failures.append(f"strict 1:1 violated: {n_pos} pos vs {n_neg} neg")
    n_neg_per = n_pos_expected // len(EXPECTED_PANEL)
    for neg in EXPECTED_PANEL:
        got = neg_by_persona.get(neg, 0)
        if got != n_neg_per:
            failures.append(f"NEG count for {neg!r}: got {got}, expected {n_neg_per}")
    for neg in neg_by_persona:
        if neg not in EXPECTED_PANEL:
            failures.append(f"unexpected NEG persona {neg!r} (panel: {list(EXPECTED_PANEL)})")
    for forbidden in (EXPECTED_NAME_A, EXPECTED_NAME_B):
        if neg_by_persona.get(forbidden, 0) != 0:
            failures.append(f"FORBIDDEN NEG rows under source {forbidden!r}")
    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"New-pair composition gate FAILED for arm={arm}:\n  - {joined}\n"
            f"POS by source: {pos_by_source}\nNEG by persona: {neg_by_persona}"
        )
    return pos_by_source, neg_by_persona


def _composition_gate_new_pair() -> None:
    """Step B: build all 3 arms of the new pair in-process and assert composition."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_538.data_build import build_arm_rows
    from explore_persona_space.experiments.issue_538.persona_registry import (
        assert_registry_resolves,
        load_persona_bank,
    )
    from explore_persona_space.experiments.issue_538.question_pool import load_question_pool

    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    for name in (EXPECTED_NAME_A, EXPECTED_NAME_B):
        if name not in persona_bank or not persona_bank[name].strip():
            raise AssertionError(
                f"source persona {name!r} does not resolve in the persona bank "
                "(navy_seal should resolve via the _AUGMENT_PERSONAS_FOR_311_PANEL path)."
            )
    log.info("Both new sources resolve: navy_seal (augment path) + french_person (on-disk bank).")

    panel = negative_panel_for_pair(EXPECTED_NAME_A, EXPECTED_NAME_B)
    if panel != EXPECTED_PANEL:
        raise AssertionError(
            f"negative_panel_for_pair({EXPECTED_NAME_A!r}, {EXPECTED_NAME_B!r}) = {panel!r}, "
            f"expected the unchanged base panel {EXPECTED_PANEL!r} (Amendment A1 resolves "
            "to the base panel for this pair — no source/panel collision)."
        )
    if set(panel) & {EXPECTED_NAME_A, EXPECTED_NAME_B}:
        raise AssertionError(f"panel {panel!r} intersects the new sources — contamination.")
    log.info("Panel resolution PASS: %s (disjoint from sources).", list(panel))

    questions = load_question_pool(n_required=400, allow_smoke_fallback=False)

    r_persona: dict[str, dict[str, str]] = {}
    for jp in sorted(R_PERSONA_DIR.glob("*.json")):
        payload = json.loads(jp.read_text())
        r_persona[payload["persona"]] = payload["responses"]
    for name in (EXPECTED_NAME_A, EXPECTED_NAME_B, *panel):
        if name not in r_persona:
            raise AssertionError(f"R_persona missing for {name!r} under {R_PERSONA_DIR}.")
        missing_qs = [q for q in questions if q not in r_persona[name]]
        if missing_qs:
            raise AssertionError(
                f"R_persona[{name!r}] covers only {len(questions) - len(missing_qs)}/"
                f"{len(questions)} pool questions (first missing: {missing_qs[0]!r})."
            )
    log.info(
        "R_persona coverage PASS: full %d-question coverage for both sources + panel.",
        len(questions),
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    expectations = {
        "joint": (N_POSITIVES_JOINT, {EXPECTED_NAME_A: 400, EXPECTED_NAME_B: 400}),
        "A_only": (N_POSITIVES_SINGLETON, {EXPECTED_NAME_A: 400}),
        "B_only": (N_POSITIVES_SINGLETON, {EXPECTED_NAME_B: 400}),
    }
    for arm, (n_pos_expected, pos_by_source_expected) in expectations.items():
        rows = build_arm_rows(
            arm=arm,
            pair_a=EXPECTED_NAME_A,
            pair_b=EXPECTED_NAME_B,
            persona_bank=persona_bank,
            questions=questions,
            r_persona=r_persona,
            tokenizer=tokenizer,
            seed=42,
        )
        pos_by_source, neg_by_persona = _assert_arm_composition(
            rows,
            arm=arm,
            n_pos_expected=n_pos_expected,
            pos_by_source_expected=pos_by_source_expected,
        )
        _assert_marker_membership(rows, arm=arm, tokenizer=tokenizer)
        log.info(
            "arm=%s PASS: POS=%s, NEG=%s, 1:1 ratio, marker in positives only.",
            arm,
            pos_by_source,
            neg_by_persona,
        )

    log.info("Step B PASS: new-pair composition gate clean across all 3 arms.")


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _ = argv  # unused; kept for symmetry with sibling scripts

    log.info("Step A: #568 pair-selection re-assert vs pre-registration")
    _reassert_pair_selection()

    log.info("Step B: new-pair composition gate (navy_seal__french_person, all 3 arms)")
    # R_persona may be absent on a fresh pod checkout; reuse the inherited
    # downloader (no-op when the dir is already populated by the #538 extras).
    _ensure_r_persona_local(R_PERSONA_DIR)
    _composition_gate_new_pair()

    log.info("ALL issue_568 preflight extras PASSED — proceed to Phase A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
