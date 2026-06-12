"""Issue #621 preflight — pinned-input prefetch + composition gate.

Forked from ``scripts/run_issue538_preflight_extras.py`` (pinned
``e6b195f81``) with the §14 duty-1 repoint:

  - The #538 byte-identity hash gate against ``issue_527/training_mixes``
    is INTENTIONALLY NOT inherited: #621's mixes legitimately differ
    (unified panel, singleton sources), so a byte-compare against the
    parent's published mixes would fail by design. What replaces it:
    content pins on the INPUTS (R_persona + question pool) + the
    composition gate below on the realized #621 mix.
  - The R_persona prefetch is upgraded to a SHA-256 mirror-identity gate
    (incident #600): every inherited file is downloaded at the pinned
    revision ``HF_TRAIN_MIX_READ_REVISION`` and sha256-asserted against
    ``EXPECTED_SHA256`` — INCLUDING files already present on the worker.

Steps:
  (A) Marker token id assert (`` ※`` == [83399]; ``<|im_end|>`` == 151645).
  (B) R_persona prefetch + sha256 pins (21 personas) → local
      ``eval_results/issue_527/R_persona/``.
  (C) Question-pool pinned fetch (sha-asserted inside question_pool) +
      400-question availability.
  (D) Composition gate: build cell (florist, seed 42) in-process and
      assert 400 POS under florist, exactly 100 NEG per unified-panel
      persona, ZERO NEG rows under any SOURCES member (the realized-
      disjointness proof, on top of the builder's own hard assert).
  (E) Persona-registry resolution gate.

CLI:
    uv run python scripts/run_issue621_preflight.py [--skip-composition-gate]
"""

# math/scientific notation in messages

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_621 import (
    BASE_MODEL,
    EXPECTED_SHA256,
    HF_DATA_REPO,
    HF_R_PATH_PREFIX,
    HF_TRAIN_MIX_READ_REVISION,
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
    SOURCES,
    UNIFIED_NEGATIVE_PANEL,
)

log = logging.getLogger("issue_621.preflight")

COMPOSITION_GATE_SOURCE = "florist"
COMPOSITION_GATE_SEED = 42
COMPOSITION_GATE_EXPECTED_POS_COUNT = 400
COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA = 100


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_r_persona_pinned(local_dir: Path) -> None:
    """Prefetch R_persona at the pinned revision + sha256-assert EVERY file.

    The assert covers files ALREADY PRESENT on the worker (a drifted local
    copy fails loud — incident #600 class), not just fresh downloads. On a
    missing/mismatched local file the pinned HF copy is downloaded,
    re-asserted, and installed.
    """
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    pinned = {
        rel: sha for rel, sha in EXPECTED_SHA256.items() if rel.startswith(f"{HF_R_PATH_PREFIX}/")
    }
    if len(pinned) < 21:
        raise AssertionError(
            f"EXPECTED_SHA256 carries only {len(pinned)} R_persona pins (<21) — "
            "the pin table is incomplete; refusing to proceed."
        )
    for rel, expected in sorted(pinned.items()):
        name = Path(rel).name
        dest = local_dir / name
        if dest.is_file():
            got = _sha256_file(dest)
            if got == expected:
                log.info("R_persona pin OK (local): %s", name)
                continue
            raise AssertionError(
                f"LOCAL R_persona file {dest} sha256={got} != pinned {expected} "
                f"(rev {HF_TRAIN_MIX_READ_REVISION}). The worker copy drifted "
                "from the planning-time verified content — refusing to train "
                "on it. Delete/inspect the local file, then re-run preflight."
            )
        hf_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=rel,
            repo_type="dataset",
            revision=HF_TRAIN_MIX_READ_REVISION,
        )
        got = _sha256_file(Path(hf_path))
        if got != expected:
            raise AssertionError(
                f"HF mirror drift: {rel} @ {HF_TRAIN_MIX_READ_REVISION} "
                f"sha256={got} != pinned {expected} (incident #600 class)."
            )
        dest.write_bytes(Path(hf_path).read_bytes())
        log.info("R_persona pin OK (downloaded): %s", name)


def _composition_gate(local_r_dir: Path) -> None:
    """Build the gate cell in-process and assert the realized composition."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_621.data_build import build_cell_rows
    from explore_persona_space.experiments.issue_621.persona_registry import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.issue_621.question_pool import (
        load_question_pool,
    )

    log.info(
        "Composition gate: building cell (%s, seed %d) in-process.",
        COMPOSITION_GATE_SOURCE,
        COMPOSITION_GATE_SEED,
    )
    persona_bank = load_persona_bank()
    questions = load_question_pool(n_required=400, allow_smoke_fallback=False)

    r_persona: dict[str, dict[str, str]] = {}
    for jp in sorted(local_r_dir.glob("*.json")):
        payload = json.loads(jp.read_text())
        r_persona[payload["persona"]] = payload["responses"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    rows = build_cell_rows(
        source=COMPOSITION_GATE_SOURCE,
        persona_bank=persona_bank,
        questions=questions,
        r_persona=r_persona,
        tokenizer=tokenizer,
        seed=COMPOSITION_GATE_SEED,
    )

    pos_by_source: dict[str, int] = {}
    neg_by_persona: dict[str, int] = {}
    for r in rows:
        tag = r.get("_arm_tag")
        if tag == "positive":
            src = r.get("_source", "<missing>")
            pos_by_source[src] = pos_by_source.get(src, 0) + 1
        elif tag == "negative":
            neg = r.get("_negative_persona", "<missing>")
            neg_by_persona[neg] = neg_by_persona.get(neg, 0) + 1
        else:
            raise AssertionError(f"row with unknown _arm_tag={tag!r}")

    failures: list[str] = []
    got_pos = pos_by_source.get(COMPOSITION_GATE_SOURCE, 0)
    if got_pos != COMPOSITION_GATE_EXPECTED_POS_COUNT:
        failures.append(
            f"POS count for {COMPOSITION_GATE_SOURCE!r}: got {got_pos}, "
            f"expected {COMPOSITION_GATE_EXPECTED_POS_COUNT}"
        )
    for src, n in pos_by_source.items():
        if src != COMPOSITION_GATE_SOURCE and n != 0:
            failures.append(f"unexpected POS rows under {src!r}: {n}")
    for neg in UNIFIED_NEGATIVE_PANEL:
        got = neg_by_persona.get(neg, 0)
        if got != COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA:
            failures.append(
                f"NEG count for {neg!r}: got {got}, expected "
                f"{COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA}"
            )
    for forbidden in SOURCES:
        got = neg_by_persona.get(forbidden, 0)
        if got != 0:
            failures.append(
                f"FORBIDDEN NEG rows under source {forbidden!r}: got {got}, "
                "expected 0 (the #527 contamination class)"
            )
    expected_keys = set(UNIFIED_NEGATIVE_PANEL)
    for neg in neg_by_persona:
        if neg not in expected_keys:
            failures.append(f"unexpected NEG persona {neg!r}")

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"Composition gate FAILED for ({COMPOSITION_GATE_SOURCE}, "
            f"seed {COMPOSITION_GATE_SEED}):\n  - {joined}\n"
            f"POS by source: {pos_by_source}\nNEG by persona: {neg_by_persona}"
        )
    log.info("Composition gate PASS: NEG=%s", neg_by_persona)


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-composition-gate",
        action="store_true",
        help="Skip step D (tokenizer-download-heavy; CPU smoke convenience only).",
    )
    ap.add_argument(
        "--r-persona-dir",
        default="eval_results/issue_527/R_persona",
        help="Local destination for the pinned R_persona prefetch.",
    )
    args = ap.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        raise AssertionError(
            "HF_TOKEN missing — pinned prefetch cannot reach the HF dataset. "
            "Source `.env` before retrying."
        )

    # §14 duty 1: surface the pinned data revision in run metadata, and
    # record that the #538 mix byte-identity gate is intentionally absent.
    log.info("HF_TRAIN_MIX_READ_REVISION=%s (pinned input revision)", HF_TRAIN_MIX_READ_REVISION)
    log.info(
        "NOTE (§14 duty 1): the #538 byte-identity training-mix gate is NOT "
        "inherited — #621 mixes legitimately differ (unified panel, singleton "
        "sources). Input pins (R_persona + question pool) + the composition "
        "gate replace it."
    )

    # (A) Marker token id assert.
    log.info("Step A: marker token id assert (` ※` -> [%d])", MARKER_ID)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(
            f"Marker token drift: encode({MARKER_TEXT!r}) -> {encoded}, expected [{MARKER_ID}]"
        )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != IM_END_ID:
        raise AssertionError(f"<|im_end|> drift: {im_end} != {IM_END_ID}")
    log.info("Step A PASS.")

    # (B) R_persona prefetch + sha pins.
    log.info("Step B: R_persona pinned prefetch (21 files, sha256-asserted)")
    _ensure_r_persona_pinned(Path(args.r_persona_dir))
    log.info("Step B PASS.")

    # (C) Question-pool pinned fetch (sha-asserted inside the loader).
    log.info("Step C: question-pool pinned fetch")
    from explore_persona_space.experiments.issue_621.question_pool import (
        load_question_pool,
    )

    qs = load_question_pool(n_required=400, allow_smoke_fallback=False)
    if len(qs) != 400:
        raise AssertionError(f"question pool returned {len(qs)} != 400")
    log.info("Step C PASS (400 questions, sha-pinned source).")

    # (E) Persona-registry resolution gate (before D so a registry gap
    # fails fast).
    log.info("Step E: persona-registry resolution gate")
    from explore_persona_space.experiments.issue_621.persona_registry import (
        assert_registry_resolves,
        load_persona_bank,
    )

    assert_registry_resolves(load_persona_bank())
    log.info("Step E PASS.")

    # (D) Composition gate on the realized #621 mix.
    if args.skip_composition_gate:
        log.warning("Step D SKIPPED (--skip-composition-gate; smoke convenience only).")
    else:
        log.info("Step D: composition gate (realized-mix disjointness proof)")
        _composition_gate(Path(args.r_persona_dir))
        log.info("Step D PASS.")

    log.info("ALL issue_621 preflight steps PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
