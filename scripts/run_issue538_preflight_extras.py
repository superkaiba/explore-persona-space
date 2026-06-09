"""Issue #538 preflight extensions on top of run_issue527_preflight.py.

Plan §4 Step 0 — extended checks specific to the issue_538 follow-up:

  (a) Hot-fix-commits ancestry assert. The two parent hot-fixes
      ``47c9466b7`` (vLLM greedy n=1) and ``8e70d0a08`` (eval prefer-local-
      adapter) live on the issue-527 branch and are inherited via the
      issue-538 branch base. Assert both are reachable from HEAD; refuse
      to advance if either is missing (the eval rig would silently regress).

  (b) Marker token id assert. ``tokenizer.encode(" ※", add_special_tokens=False)
      == [83399]`` — inherited from #527 preflight but repeated here as the
      first issue_538-specific gate so a tokenizer drift is caught loud.

  (c) R_persona presence + HF fallback. Verify
      ``eval_results/issue_527/R_persona/`` exists and contains at least 19
      JSONs. If missing, download from HF dataset
      ``superkaiba1/explore-persona-space-data`` at revision
      ``e6e163ce2a58108cc2c2d530f5f0ea9ef4542f65`` path
      ``issue_527/R_persona/``.

  (d) Hash gate (the core regression guard). Regenerate the cell
      ``florist__medical_doctor__A_only__seed42`` with the canonical
      ``build_arm_rows(...)`` seeded RNG and sha256-compare the resulting
      JSONL against the HF-published copy under
      ``issue_527/training_mixes/`` at revision ``e6e163ce...``. Fail loud
      with a 5-line head diff on mismatch — that means R_persona or
      persona_bank drifted and the byte-identical-determinism contract
      is broken. The gate cell is pair-1 BY DESIGN: pair-1's panel has no
      overlap so the task #538 per-pair-panel fix is a provable no-op
      for it; pair-2 mixes deliberately diverge from #527 per the panel
      fix (see task #538 21:27Z ``epm:concern-raised`` marker), so the
      pair-2 cell gets a separate composition gate (Step G) instead of a
      hash compare.

  (g) Pair-2 composition gate (task #538 fix verification). Build cell
      ``librarian__police_officer__A_only__seed42`` IN-PROCESS (no HF
      compare — pair-2 deliberately diverges from #527) and assert:
        - 400 POS rows under ``librarian``
        - exactly 100 NEG rows for each of
          {assistant, kindergarten_teacher, programmer, chef}
        - 0 NEG rows for ``librarian`` and ``police_officer``
      This is the executable proof the #527 contamination (same persona
      trained POS + NEG 4:1 in the same cell) is gone.

  (e) Pair-selection file presence (inherited from #527 — required for
      train dispatcher to enumerate pairs).

  (f) Adapter gauge readiness — config-only. We don't have an adapter yet
      at preflight, so the gauge assert per plan §6 lives in the
      ``run_issue538_eval.py`` shift_extract path right after PEFT load.
      Logged here so the failure surface is documented in one place.

CLI:
    uv run python scripts/run_issue538_preflight_extras.py
"""

# math/scientific notation in messages

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_538 import (
    BASE_MODEL,
    HF_DATA_REPO,
    HF_R_PATH_PREFIX,
    HF_TRAIN_MIX_READ_PATH_PREFIX,
    HF_TRAIN_MIX_READ_REVISION,
    MARKER_ID,
    MARKER_TEXT,
)

log = logging.getLogger("issue_538.preflight_extras")

# Inherited hot-fix commits from #527 (plan §4 Branch base).
HOTFIX_COMMITS = ("47c9466b7", "8e70d0a08")

# Hash-gate cell (pair-1, no panel overlap → byte-identical vs #527).
HASH_GATE_CELL_SLUG = "florist__medical_doctor__A_only__seed42"

# Pair-2 composition gate cell (task #538 fix — pair-2 deliberately diverges
# from #527 via the per-pair panel; verified by in-process composition, NOT
# a hash compare).
COMPOSITION_GATE_CELL_SLUG = "librarian__police_officer__A_only__seed42"
COMPOSITION_GATE_EXPECTED_POS_PERSONA = "librarian"
COMPOSITION_GATE_EXPECTED_POS_COUNT = 400
COMPOSITION_GATE_EXPECTED_NEG_PANEL = (
    "assistant",
    "kindergarten_teacher",
    "programmer",
    "chef",
)
COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA = 100
COMPOSITION_GATE_FORBIDDEN_NEG_PERSONAS = ("librarian", "police_officer")


def _git_ancestor_assert(commit: str) -> None:
    """Assert ``commit`` is reachable from HEAD on the current branch."""
    try:
        subprocess.check_output(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        raise AssertionError(
            f"Hot-fix commit {commit} is NOT reachable from HEAD "
            f"(branch={branch}, head={head}). The issue-538 branch MUST be "
            f"cut from issue-527 per plan §4 Branch base; the eval rig depends "
            f"on this hot-fix."
        ) from None


def _ensure_r_persona_local(local_dir: Path) -> None:
    """Verify the local R_persona dir exists; download from HF if missing."""
    if local_dir.is_dir():
        n = len(list(local_dir.glob("*.json")))
        if n >= 19:
            log.info("R_persona local dir OK at %s (%d JSONs).", local_dir, n)
            return
        log.warning(
            "R_persona local dir %s has only %d JSONs (<19); re-downloading from HF.",
            local_dir,
            n,
        )

    log.info(
        "R_persona local dir missing or short; downloading from HF dataset %s/%s at revision %s",
        HF_DATA_REPO,
        HF_R_PATH_PREFIX,
        HF_TRAIN_MIX_READ_REVISION,
    )
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(HF_DATA_REPO, repo_type="dataset", revision=HF_TRAIN_MIX_READ_REVISION)
    r_files = [f for f in files if f.startswith(f"{HF_R_PATH_PREFIX}/") and f.endswith(".json")]
    if not r_files:
        raise RuntimeError(
            f"No R_persona JSONs found under HF dataset "
            f"{HF_DATA_REPO}/{HF_R_PATH_PREFIX}/ at revision "
            f"{HF_TRAIN_MIX_READ_REVISION}. Cannot proceed."
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    for f in r_files:
        local_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f,
            repo_type="dataset",
            revision=HF_TRAIN_MIX_READ_REVISION,
        )
        # Copy to local_dir (hf_hub_download returns a symlink into the cache).
        dest = local_dir / Path(f).name
        if not dest.is_file():
            dest.write_bytes(Path(local_path).read_bytes())
    n_after = len(list(local_dir.glob("*.json")))
    log.info("Downloaded %d R_persona JSONs to %s", n_after, local_dir)
    if n_after < 19:
        raise RuntimeError(
            f"After download, R_persona dir has {n_after} JSONs (<19). HF copy may be incomplete."
        )


def _hash_gate(persona_bank_path: Path) -> None:
    """Regenerate the gate cell + sha256-compare to the HF-published copy."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_538.data_build import (
        build_arm_rows,
        write_rows_jsonl,
    )
    from explore_persona_space.experiments.issue_538.persona_registry import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.issue_538.question_pool import (
        load_question_pool,
    )

    log.info(
        "Hash gate: regenerating cell %s and sha256-comparing to HF copy.",
        HASH_GATE_CELL_SLUG,
    )

    # Resolve cell-slug parts.
    pair_a, pair_b, arm, seed_str = HASH_GATE_CELL_SLUG.rsplit("__", 3)
    seed = int(seed_str.removeprefix("seed"))

    persona_bank = load_persona_bank()
    questions = load_question_pool(n_required=400, allow_smoke_fallback=False)

    # Load R_persona from the just-confirmed local dir.
    r_persona_dir = Path("eval_results/issue_527/R_persona")
    r_persona: dict[str, dict[str, str]] = {}
    for jp in sorted(r_persona_dir.glob("*.json")):
        payload = json.loads(jp.read_text())
        r_persona[payload["persona"]] = payload["responses"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Regenerate.
    rows = build_arm_rows(
        arm=arm,
        pair_a=pair_a,
        pair_b=pair_b,
        persona_bank=persona_bank,
        questions=questions,
        r_persona=r_persona,
        tokenizer=tokenizer,
        seed=seed,
    )
    out_path = Path("/tmp") / f"{HASH_GATE_CELL_SLUG}__hashgate.jsonl"
    write_rows_jsonl(rows, out_path)
    local_bytes = out_path.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    log.info("Local regen sha256=%s (rows=%d)", local_sha, len(rows))

    # Download the HF copy at the pinned revision.
    hf_remote_path = f"{HF_TRAIN_MIX_READ_PATH_PREFIX}/{HASH_GATE_CELL_SLUG}.jsonl"
    log.info(
        "Downloading HF copy from %s/%s at revision %s",
        HF_DATA_REPO,
        hf_remote_path,
        HF_TRAIN_MIX_READ_REVISION,
    )
    hf_path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=hf_remote_path,
        repo_type="dataset",
        revision=HF_TRAIN_MIX_READ_REVISION,
    )
    hf_bytes = Path(hf_path).read_bytes()
    hf_sha = hashlib.sha256(hf_bytes).hexdigest()
    log.info("HF sha256=%s", hf_sha)

    if local_sha != hf_sha:
        # Show a 5-line head diff for diagnosis (per plan §4 Step 0).
        local_head = local_bytes.decode("utf-8", errors="replace").splitlines()[:5]
        hf_head = hf_bytes.decode("utf-8", errors="replace").splitlines()[:5]
        raise AssertionError(
            f"Hash gate FAILED for {HASH_GATE_CELL_SLUG}: local sha256={local_sha} "
            f"vs HF sha256={hf_sha}. Either R_persona drifted, persona_bank "
            f"drifted, or build_arm_rows changed. Local head:\n"
            + "\n".join(f"  L: {line}" for line in local_head)
            + "\nHF head:\n"
            + "\n".join(f"  H: {line}" for line in hf_head)
        )
    log.info("Hash gate PASS: regenerated cell sha256 matches HF copy byte-identically.")


def _composition_gate() -> None:
    """Pair-2 composition gate (task #538 fix — pure local, no HF compare).

    Build cell ``librarian__police_officer__A_only__seed42`` in-process via
    ``build_arm_rows`` and assert the EXPECTED per-pair composition:

      - 400 POS rows under ``librarian`` (the realized A source)
      - 100 NEG rows for each of
        {assistant, kindergarten_teacher, programmer, chef}
      - 0 NEG rows for ``librarian`` and ``police_officer`` (the sources
        must NEVER appear as a negative in their own cell — that is the
        #527 contamination this fix removes)

    Pair-2 diverges from #527's training mix by design (the panel fix is
    the deliberate divergence), so this gate replaces the hash compare
    with a pure executable composition check.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_538.data_build import (
        build_arm_rows,
    )
    from explore_persona_space.experiments.issue_538.persona_registry import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.issue_538.question_pool import (
        load_question_pool,
    )

    log.info(
        "Composition gate: building cell %s in-process to verify panel fix.",
        COMPOSITION_GATE_CELL_SLUG,
    )

    pair_a, pair_b, arm, seed_str = COMPOSITION_GATE_CELL_SLUG.rsplit("__", 3)
    seed = int(seed_str.removeprefix("seed"))

    persona_bank = load_persona_bank()
    questions = load_question_pool(n_required=400, allow_smoke_fallback=False)

    r_persona_dir = Path("eval_results/issue_527/R_persona")
    r_persona: dict[str, dict[str, str]] = {}
    for jp in sorted(r_persona_dir.glob("*.json")):
        payload = json.loads(jp.read_text())
        r_persona[payload["persona"]] = payload["responses"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    rows = build_arm_rows(
        arm=arm,
        pair_a=pair_a,
        pair_b=pair_b,
        persona_bank=persona_bank,
        questions=questions,
        r_persona=r_persona,
        tokenizer=tokenizer,
        seed=seed,
    )

    # Tally positives by source and negatives by persona.
    pos_count_by_source: dict[str, int] = {}
    neg_count_by_persona: dict[str, int] = {}
    for r in rows:
        tag = r.get("_arm_tag")
        if tag == "positive":
            src = r.get("_source", "<missing>")
            pos_count_by_source[src] = pos_count_by_source.get(src, 0) + 1
        elif tag == "negative":
            neg = r.get("_negative_persona", "<missing>")
            neg_count_by_persona[neg] = neg_count_by_persona.get(neg, 0) + 1
        else:
            raise AssertionError(
                f"row with unknown _arm_tag={tag!r} in {COMPOSITION_GATE_CELL_SLUG}"
            )

    failures: list[str] = []

    # (1) Positive-side: exactly N rows under the expected persona, none elsewhere.
    got_pos = pos_count_by_source.get(COMPOSITION_GATE_EXPECTED_POS_PERSONA, 0)
    if got_pos != COMPOSITION_GATE_EXPECTED_POS_COUNT:
        failures.append(
            f"POS count for {COMPOSITION_GATE_EXPECTED_POS_PERSONA!r}: "
            f"got {got_pos}, expected {COMPOSITION_GATE_EXPECTED_POS_COUNT}"
        )
    for src, n in pos_count_by_source.items():
        if src != COMPOSITION_GATE_EXPECTED_POS_PERSONA and n != 0:
            failures.append(
                f"unexpected POS rows under {src!r}: {n} (only "
                f"{COMPOSITION_GATE_EXPECTED_POS_PERSONA!r} should carry positives in A_only)"
            )

    # (2) Negative-side: exactly N per expected panel member.
    for neg in COMPOSITION_GATE_EXPECTED_NEG_PANEL:
        got = neg_count_by_persona.get(neg, 0)
        if got != COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA:
            failures.append(
                f"NEG count for {neg!r}: got {got}, expected "
                f"{COMPOSITION_GATE_EXPECTED_NEG_PER_PERSONA}"
            )

    # (3) Negative-side: ZERO rows for sources (the #527 contamination check).
    for forbidden in COMPOSITION_GATE_FORBIDDEN_NEG_PERSONAS:
        got = neg_count_by_persona.get(forbidden, 0)
        if got != 0:
            failures.append(
                f"FORBIDDEN NEG rows under source {forbidden!r}: got {got}, expected 0 "
                f"(this is the #527 contamination the panel fix removes)"
            )

    # (4) Any unexpected negative-persona key.
    expected_keys = set(COMPOSITION_GATE_EXPECTED_NEG_PANEL)
    for neg in neg_count_by_persona:
        if neg not in expected_keys:
            failures.append(
                f"unexpected NEG persona {neg!r} (panel for pair-2 should be "
                f"{list(COMPOSITION_GATE_EXPECTED_NEG_PANEL)})"
            )

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"Composition gate FAILED for {COMPOSITION_GATE_CELL_SLUG}:\n  - {joined}\n"
            f"POS by source: {pos_count_by_source}\n"
            f"NEG by persona: {neg_count_by_persona}"
        )
    log.info(
        "Composition gate PASS: %s composition matches the task #538 per-pair panel. "
        "POS={%s: %d}; NEG=%s",
        COMPOSITION_GATE_CELL_SLUG,
        COMPOSITION_GATE_EXPECTED_POS_PERSONA,
        COMPOSITION_GATE_EXPECTED_POS_COUNT,
        neg_count_by_persona,
    )


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _ = argv  # unused; kept for symmetry with sibling scripts

    # (a) Hot-fix commits ancestry assert.
    log.info("Step A: hot-fix-commits ancestry assert (%s)", HOTFIX_COMMITS)
    for c in HOTFIX_COMMITS:
        _git_ancestor_assert(c)
    log.info("Step A PASS.")

    # (b) Marker token id assert (defense-in-depth — also in run_issue527_preflight.py).
    log.info("Step B: marker token id assert (` ※` should encode to [%d])", MARKER_ID)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(
            f"Marker token drift: encode({MARKER_TEXT!r}) -> {encoded}, expected [{MARKER_ID}]"
        )
    log.info("Step B PASS.")

    # (c) R_persona presence + HF fallback.
    log.info("Step C: R_persona presence + HF fallback")
    r_persona_dir = Path("eval_results/issue_527/R_persona")
    _ensure_r_persona_local(r_persona_dir)
    log.info("Step C PASS.")

    # (e) Pair-selection presence (read pre hash gate so a missing
    # pair_selection.json fails fast — required for train dispatcher).
    log.info("Step E: pair_selection.json presence")
    ps = Path("eval_results/issue_527/pair_selection.json")
    if not ps.is_file():
        raise AssertionError(
            f"pair_selection.json missing at {ps}; INHERITED from #527 per plan §4 Inputs. "
            "The issue-538 branch was cut from issue-527 — if missing, the branch base "
            "is wrong."
        )
    log.info("Step E PASS.")

    # (d) Hash gate — the core regression guard.
    log.info("Step D: hash gate against HF dataset issue_527/training_mixes/")
    if not os.environ.get("HF_TOKEN"):
        raise AssertionError(
            "HF_TOKEN missing — hash gate cannot reach the HF dataset. Set HF_TOKEN "
            "(or source `.env`) before retrying."
        )
    _hash_gate(persona_bank_path=Path("data/issue_472/persona_bank.json"))

    # (g) Pair-2 composition gate (task #538 fix — pure local, no HF compare).
    log.info("Step G: pair-2 composition gate (task #538 per-pair-panel fix)")
    _composition_gate()
    log.info("Step G PASS.")

    # (f) Documented stub — gauge assert lives in the eval shift_extract path.
    log.info(
        "Step F: gauge assert is documented to run inside run_issue538_eval.py "
        "shift_extract path (post PEFT load); no preflight forward needed."
    )

    log.info("ALL issue_538 preflight extras PASSED — proceed to Phase A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
