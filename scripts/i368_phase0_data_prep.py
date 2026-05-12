#!/usr/bin/env python3
"""Phase 0 data prep for issue #368.

Implements the three Phase-0 stages from the approved plan:

  0.0  Panel-recoverability gate (R1 + R7).
       Filter ``eval_results/issue_207/js_gentle/base_model_generations.json::
       system_prompts`` to records with ``id in csv_ids``; assert exactly 32
       records remain (the 32 panel ids in the regression CSV). Optionally
       SHA256-verify the canonical ``{test_id: prompt_text}`` dict against
       ``MANIFEST.json::eval_panel_sha256`` if a manifest exists.

  0.1a Sonnet paraphrase generation for the 4 non-persona triggers.
       Reads ``data/i181_non_persona/triggers.json``, generates 4 paraphrases
       per trigger via Anthropic Claude Sonnet (seed=42 in prompt; re-rolls
       on failure with seed+=1), validates with sentence-transformer cosine
       (pairwise ≤0.95, vs-verbatim ≥0.55), writes to
       ``data/i181_non_persona/instructions/{T_task,T_instruction,T_context,
       T_format}.json``.

  0.1b Universal helpful-assistant negative set.
       Writes the hand-authored 5-paraphrase file to
       ``data/assistant_axis/instructions/_helpful_assistant_negset.json``.

  0.1c Regenerate the 10 non-baseline persona instruction files (T11 Option A).
       Backs up any existing ``{persona}.json`` into ``_backup/``, then
       generates 4 paraphrases per persona seeded by
       ``personas.py::PERSONAS``, mirroring the trigger paraphrase pipeline.

Writes nothing to git outside the worktree. The trigger / persona JSON files
themselves are then committed at commit-time by the caller.

Schema for every output instruction file (4 trigger files + 10 persona files +
1 universal-neg file) matches the existing 5-paraphrase ``{"instruction":
[{"pos": "..."} ...]}`` schema used across ``data/assistant_axis/``::

    {
      "instruction": [
        {"pos": "<verbatim seed>"},
        {"pos": "<paraphrase 1>"},
        {"pos": "<paraphrase 2>"},
        {"pos": "<paraphrase 3>"},
        {"pos": "<paraphrase 4>"}
      ]
    }

Usage::

    # Run all three stages
    uv run python scripts/i368_phase0_data_prep.py

    # Just verify the panel gate (no Sonnet calls)
    uv run python scripts/i368_phase0_data_prep.py --gate-only

    # Just generate paraphrases (assumes gate already passed)
    uv run python scripts/i368_phase0_data_prep.py --skip-gate

    # Smoke test: shrink panel to 1 trigger + 1 persona, no API calls
    uv run python scripts/i368_phase0_data_prep.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Project paths (resolve absolute so the script works regardless of cwd)
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.personas import PERSONAS  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

REGRESSION_CSV = REPO_ROOT / "eval_results" / "issue_207" / "js_gentle" / "regression_data.csv"
PANEL_GENERATIONS_JSON = (
    REPO_ROOT / "eval_results" / "issue_207" / "js_gentle" / "base_model_generations.json"
)
PANEL_MANIFEST_JSON = REPO_ROOT / "eval_results" / "issue_207" / "js_gentle" / "MANIFEST.json"
PANEL_FALLBACK_WORKTREE = (
    REPO_ROOT.parent / "issue-274" / "data" / "i181_non_persona" / "eval_panel.json"
)

TRIGGERS_JSON = REPO_ROOT / "data" / "i181_non_persona" / "triggers.json"
TRIGGER_OUT_DIR = REPO_ROOT / "data" / "i181_non_persona" / "instructions"

ASSISTANT_AXIS_DIR = REPO_ROOT / "data" / "assistant_axis"
ASSISTANT_INSTRUCTIONS_DIR = ASSISTANT_AXIS_DIR / "instructions"
ASSISTANT_BACKUP_DIR = ASSISTANT_INSTRUCTIONS_DIR / "_backup"
NEGSET_JSON = ASSISTANT_INSTRUCTIONS_DIR / "_helpful_assistant_negset.json"

# Universal helpful-assistant negative side (hand-written; 5 paraphrases)
HELPFUL_NEG_PARAPHRASES: list[str] = [
    "You are a helpful assistant.",
    "You are a helpful AI assistant.",
    "You are an assistant designed to be helpful and informative.",
    "You are a helpful, knowledgeable assistant who provides clear answers.",
    "You are an AI assistant focused on being helpful and useful.",
]

NON_BASELINE_PERSONAS: list[str] = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "villain",
    "zelthari_scholar",
]

TRIGGER_NAMES: list[str] = ["T_task", "T_instruction", "T_context", "T_format"]
TRIGGER_FAMILY: dict[str, str] = {
    "T_task": "task",
    "T_instruction": "instruction",
    "T_context": "context",
    "T_format": "format",
}

CLAUDE_SONNET_MODEL = "claude-sonnet-4-5-20250929"

# Sentence-transformer similarity thresholds (plan §4.0.1)
MAX_PAIRWISE_COSINE = 0.95
MIN_VS_VERBATIM_COSINE = 0.55


# ── Phase 0.0 — panel-recoverability gate (R1 + R7) ─────────────────────────


def _canonical_panel_json(panel_strings: dict[str, str]) -> str:
    """Canonicalize the panel mapping for SHA256 hashing.

    Sort keys, no whitespace, UTF-8. Used by R7 manifest check.
    """
    return json.dumps(panel_strings, sort_keys=True, separators=(",", ":"))


def _csv_test_ids() -> set[str]:
    if not REGRESSION_CSV.exists():
        raise FileNotFoundError(
            f"regression_data.csv not found at {REGRESSION_CSV}. "
            "Phase 0.0 requires #207's regression CSV on disk."
        )
    with open(REGRESSION_CSV) as f:
        return {r["test_id"] for r in csv.DictReader(f)}


def _load_panel_strings_from_local() -> dict[str, str] | None:
    """Try the local base_model_generations.json. Returns None if absent."""
    if not PANEL_GENERATIONS_JSON.exists():
        return None
    with open(PANEL_GENERATIONS_JSON) as f:
        gen = json.load(f)
    csv_ids = _csv_test_ids()
    # R1 fix: filter to source==panel records BEFORE comparing to csv_ids,
    # OR equivalently filter to records whose id is in csv_ids.
    panel_strings = {
        sp["id"]: sp["text"] for sp in gen.get("system_prompts", []) if sp["id"] in csv_ids
    }
    if set(panel_strings.keys()) != csv_ids:
        missing = csv_ids - set(panel_strings.keys())
        extra = set(panel_strings.keys()) - csv_ids
        raise RuntimeError(
            f"Phase 0.0 gate FAILED: panel id mismatch.\n"
            f"  missing from base_model_generations: {sorted(missing)}\n"
            f"  unexpected ids:                      {sorted(extra)}"
        )
    if len(panel_strings) != 32:
        raise RuntimeError(
            f"Phase 0.0 gate FAILED: expected 32 panel records, got {len(panel_strings)}"
        )
    return panel_strings


def _load_panel_strings_from_fallback() -> dict[str, str] | None:
    """Try the issue-274 worktree fallback. Returns None if absent.

    Format expected: ``{test_id: prompt_text}`` dict OR a list of
    ``{"id": ..., "text": ...}`` records.
    """
    if not PANEL_FALLBACK_WORKTREE.exists():
        return None
    with open(PANEL_FALLBACK_WORKTREE) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    if isinstance(raw, list):
        return {r["id"]: r["text"] for r in raw}
    raise RuntimeError(
        f"Phase 0.0 gate FAILED: fallback {PANEL_FALLBACK_WORKTREE} has "
        f"unexpected shape {type(raw)}"
    )


def _hash_check_against_manifest(panel_strings: dict[str, str]) -> str | None:
    """R7 SHA256 verification.

    Returns a string describing the check ('matched' / 'no manifest') or
    raises if a manifest exists AND hashes disagree.
    """
    sha = hashlib.sha256(_canonical_panel_json(panel_strings).encode("utf-8")).hexdigest()
    if not PANEL_MANIFEST_JSON.exists():
        return f"no manifest (sha256={sha[:12]}...); skipping R7 check"
    with open(PANEL_MANIFEST_JSON) as f:
        manifest = json.load(f)
    expected = manifest.get("eval_panel_sha256")
    if not expected:
        return f"manifest exists but no eval_panel_sha256 key (sha256={sha[:12]}...)"
    if expected != sha:
        raise RuntimeError(
            f"Phase 0.0 gate FAILED: SHA256 mismatch.\n"
            f"  expected: {expected}\n"
            f"  observed: {sha}\n"
            f"Refusing to proceed — re-running build_i181_data.py --step "
            f"panel-only would corrupt the system_prompt ↔ marker_rate "
            f"alignment because 16 of the 32 prompts are LLM-generated "
            f"non-deterministically."
        )
    return f"R7 SHA256 matched manifest ({sha[:12]}...)"


def run_phase00_gate(verbose: bool = True) -> dict[str, str]:
    """Recover the 32 panel system-prompt strings.

    Returns the ``{test_id: prompt_text}`` dict. Raises on any gate failure.
    The dict is NOT persisted to disk — callers (projection script) re-call
    this function or rely on base_model_generations.json being present on
    the pod.
    """
    if verbose:
        print(f"[Phase 0.0] Loading panel strings from {PANEL_GENERATIONS_JSON.name}...")

    panel_strings = _load_panel_strings_from_local()
    source = "local"
    if panel_strings is None:
        if verbose:
            print(
                f"  Local base_model_generations.json absent; trying worktree "
                f"fallback {PANEL_FALLBACK_WORKTREE}..."
            )
        panel_strings = _load_panel_strings_from_fallback()
        source = "fallback-worktree"
    if panel_strings is None:
        raise RuntimeError(
            "Phase 0.0 gate FAILED: no panel source found. Tried:\n"
            f"  - {PANEL_GENERATIONS_JSON}\n"
            f"  - {PANEL_FALLBACK_WORKTREE}\n"
            "Halt + escalate per plan §4.1.2 fallback chain. Re-running "
            "build_i181_data.py --step panel-only is NOT allowed (16 of "
            "the 32 prompts are LLM-generated non-deterministically)."
        )

    # Sanity: the dict must cover exactly the regression CSV's test_ids.
    csv_ids = _csv_test_ids()
    if set(panel_strings.keys()) != csv_ids:
        raise RuntimeError(
            f"Phase 0.0 gate FAILED ({source}): panel id mismatch.\n"
            f"  CSV has {len(csv_ids)} ids, source has {len(panel_strings)} ids;\n"
            f"  symmetric diff: {sorted(csv_ids ^ set(panel_strings.keys()))}"
        )

    hash_status = _hash_check_against_manifest(panel_strings)
    if verbose:
        print(f"  [{source}] {len(panel_strings)} panel ids recovered. {hash_status}")
    return panel_strings


# ── Phase 0.1a — Sonnet paraphrase generation ───────────────────────────────


def _build_paraphrase_user_message(verbatim: str, family: str) -> str:
    """The user-message prompt for Sonnet (4 paraphrases per call)."""
    return (
        "You are paraphrasing a system prompt that defines an agent's working "
        "context.\n"
        "Preserve the semantic content (the same task / instruction / context "
        "/ format directive) but vary the surface form. Do NOT add or remove "
        'the [ZLT] marker. Do NOT introduce persona language ("You are '
        'a..."). Stay in the same register.\n\n'
        f"Trigger verbatim: {verbatim}\n"
        f"Family: {family}\n\n"
        "Return 4 paraphrases as a JSON list of strings, nothing else."
    )


def _build_persona_paraphrase_message(verbatim: str, persona_name: str) -> str:
    """The user-message prompt for persona paraphrases.

    Personas DO open with persona language ("You are a..."), unlike triggers.
    """
    return (
        "You are paraphrasing a persona system prompt that defines a "
        "character / role for an AI assistant.\n"
        "Preserve the same character / role (same profession, traits, "
        "speaking style) but vary the surface form across paraphrases.\n"
        'Each paraphrase MUST open with persona language ("You are a..." or '
        '"You\'re a..." or similar). Stay in the same register.\n\n'
        f"Persona name: {persona_name}\n"
        f"Verbatim seed: {verbatim}\n\n"
        "Return 4 paraphrases as a JSON list of strings, nothing else."
    )


_JSON_LIST_RE = re.compile(r"\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]", re.DOTALL)


def _parse_paraphrase_response(text: str, n_expected: int = 4) -> list[str]:
    """Parse Sonnet's response into a list of strings.

    Handles common deviations: fenced ```json blocks, leading/trailing prose.
    Raises if it cannot find a JSON list of strings with ``n_expected`` items.
    """
    # Strip code-fence wrappers if present.
    candidate = text.strip()
    if candidate.startswith("```"):
        # Remove first line (```json or ```) and trailing ```
        lines = candidate.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    # Try direct parse first.
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as err:
        # Find a `[...]` block.
        m = _JSON_LIST_RE.search(text)
        if not m:
            raise ValueError(f"No JSON list found in Sonnet response:\n{text[:500]}") from err
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list):
        raise ValueError(f"Parsed JSON is not a list: {type(parsed).__name__}")
    parsed = [p.strip() for p in parsed if isinstance(p, str) and p.strip()]
    if len(parsed) != n_expected:
        raise ValueError(f"Expected {n_expected} paraphrases, got {len(parsed)}: {parsed}")
    return parsed


def _call_sonnet_for_paraphrases(
    user_message: str,
    *,
    n_expected: int = 4,
    max_attempts: int = 4,
    api_key: str | None = None,
    seed_hint: int = 42,
) -> list[str]:
    """Synchronous Sonnet call returning ``n_expected`` paraphrases.

    Re-rolls up to ``max_attempts`` times on parse failure or empty content,
    bumping ``seed_hint`` each time so the prompt content varies slightly.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        seed_line = (
            f"\n\n(Generation hint: seed={seed_hint + attempt}. Vary surface "
            "form across paraphrases.)"
        )
        try:
            msg = client.messages.create(
                model=CLAUDE_SONNET_MODEL,
                max_tokens=800,
                temperature=0.7,
                messages=[{"role": "user", "content": user_message + seed_line}],
            )
            text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                raise ValueError("Sonnet returned no text content.")
            text = "\n".join(text_blocks).strip()
            return _parse_paraphrase_response(text, n_expected=n_expected)
        except Exception as e:  # parse / API errors — retry
            last_err = e
            time.sleep(0.5 * (attempt + 1))
            continue
    raise RuntimeError(
        f"Sonnet paraphrase generation failed after {max_attempts} attempts: {last_err!r}"
    )


# ── Sentence-transformer similarity gate (R5) ───────────────────────────────


def _maybe_load_sentence_transformer():
    """Load all-MiniLM-L6-v2; return None if sentence-transformers unavailable.

    Returning None lets the smoke-test path proceed without the optional dep.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _validate_paraphrase_similarity(
    paraphrases: list[str], verbatim: str, model, label: str
) -> tuple[bool, str]:
    """Check pairwise + vs-verbatim cosine constraints.

    Returns (ok, diagnostic_string). Plan §4.0.1: max pairwise ≤ 0.95, vs
    verbatim ≥ 0.55.
    """
    if model is None:
        return True, "sentence-transformers unavailable; skipped similarity gate"

    embs = model.encode([verbatim, *paraphrases], normalize_embeddings=True)
    verbatim_emb = embs[0]
    para_embs = embs[1:]

    vs_verb = (para_embs @ verbatim_emb).tolist()
    too_far = [i for i, c in enumerate(vs_verb) if c < MIN_VS_VERBATIM_COSINE]
    if too_far:
        return (
            False,
            f"{label}: paraphrase(s) {too_far} cosine-vs-verbatim "
            f"{[round(vs_verb[i], 3) for i in too_far]} < {MIN_VS_VERBATIM_COSINE}",
        )

    pairwise = para_embs @ para_embs.T
    n = len(paraphrases)
    max_pair = -1.0
    max_ij = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if pairwise[i, j] > max_pair:
                max_pair = float(pairwise[i, j])
                max_ij = (i, j)
    if max_pair > MAX_PAIRWISE_COSINE:
        return (
            False,
            f"{label}: max pairwise cosine {round(max_pair, 3)} between paraphrases "
            f"{max_ij} > {MAX_PAIRWISE_COSINE}",
        )

    # Convert numpy floats to native floats for the diagnostic string.
    return True, (
        f"{label}: ok (min-vs-verbatim={round(min(vs_verb), 3)}, max-pairwise={round(max_pair, 3)})"
    )


def _generate_paraphrase_set(
    *,
    verbatim: str,
    user_message_builder,
    label: str,
    similarity_model,
    smoke_test: bool,
    api_key: str | None = None,
) -> list[str]:
    """Generate + validate 4 paraphrases for one trait.

    On similarity-gate failure, re-rolls with a fresh Sonnet call (seed bump),
    up to 3 times.
    """
    if smoke_test:
        # In smoke-test we don't call Sonnet — return deterministic stubs.
        return [f"{verbatim} (paraphrase smoke-{i + 1})" for i in range(4)]

    last_diag = ""
    for outer_attempt in range(3):
        paraphrases = _call_sonnet_for_paraphrases(
            user_message_builder(verbatim, label),
            seed_hint=42 + outer_attempt,
            api_key=api_key,
        )
        ok, diag = _validate_paraphrase_similarity(paraphrases, verbatim, similarity_model, label)
        if ok:
            print(f"  [{label}] {diag}")
            return paraphrases
        last_diag = diag
        print(f"  [{label}] retry {outer_attempt + 1}/3: {diag}")
    raise RuntimeError(
        f"Paraphrase similarity gate failed after 3 rerolls for {label}: {last_diag}"
    )


def _write_instruction_file(out_path: Path, verbatim: str, paraphrases: list[str]) -> None:
    payload = {"instruction": [{"pos": verbatim}] + [{"pos": p} for p in paraphrases]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    assert len(payload["instruction"]) == 5, "Expected 5 entries in instruction file."


def generate_trigger_paraphrases(smoke_test: bool = False, api_key: str | None = None) -> None:
    """Phase 0.1a: 4 trigger files in ``data/i181_non_persona/instructions/``."""
    with open(TRIGGERS_JSON) as f:
        triggers = json.load(f)["triggers"]

    similarity_model = None if smoke_test else _maybe_load_sentence_transformer()
    print(f"[Phase 0.1a] Generating trigger paraphrases (smoke_test={smoke_test})")

    target_triggers = TRIGGER_NAMES if not smoke_test else TRIGGER_NAMES[:1]

    for trigger_name in target_triggers:
        verbatim = triggers[trigger_name]
        family = TRIGGER_FAMILY[trigger_name]
        out_path = TRIGGER_OUT_DIR / f"{trigger_name}.json"
        if out_path.exists() and not smoke_test:
            print(f"  [{trigger_name}] exists; skipping (delete to regenerate)")
            continue
        paraphrases = _generate_paraphrase_set(
            verbatim=verbatim,
            user_message_builder=lambda v, _label, fam=family: _build_paraphrase_user_message(
                v, fam
            ),
            label=trigger_name,
            similarity_model=similarity_model,
            smoke_test=smoke_test,
            api_key=api_key,
        )
        _write_instruction_file(out_path, verbatim, paraphrases)
        print(f"  [{trigger_name}] wrote {out_path.relative_to(REPO_ROOT)}")


def write_universal_negset() -> None:
    """Phase 0.1b: hand-authored helpful-assistant negative set."""
    NEGSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"instruction": [{"pos": p} for p in HELPFUL_NEG_PARAPHRASES]}
    with open(NEGSET_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[Phase 0.1b] Wrote {NEGSET_JSON.relative_to(REPO_ROOT)} "
        f"({len(HELPFUL_NEG_PARAPHRASES)} paraphrases)"
    )


def regenerate_persona_instructions(smoke_test: bool = False, api_key: str | None = None) -> None:
    """Phase 0.1c (T11 Option A): regenerate all 10 persona instruction files.

    Backup existing files to ``_backup/``. Generate via Sonnet from
    ``personas.py::PERSONAS`` seeds.
    """
    similarity_model = None if smoke_test else _maybe_load_sentence_transformer()
    ASSISTANT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 0.1c] Regenerating persona instructions (smoke_test={smoke_test})")
    target_personas = NON_BASELINE_PERSONAS if not smoke_test else NON_BASELINE_PERSONAS[:1]

    for persona_name in target_personas:
        verbatim = PERSONAS[persona_name]
        out_path = ASSISTANT_INSTRUCTIONS_DIR / f"{persona_name}.json"
        if out_path.exists():
            backup_path = ASSISTANT_BACKUP_DIR / f"{persona_name}.json"
            shutil.copy2(out_path, backup_path)
            print(f"  [{persona_name}] backed up existing -> {backup_path.relative_to(REPO_ROOT)}")
        if smoke_test and out_path.exists():
            # In smoke-test, skip rewriting existing files.
            print(f"  [{persona_name}] smoke-test: keeping existing")
            continue
        paraphrases = _generate_paraphrase_set(
            verbatim=verbatim,
            user_message_builder=_build_persona_paraphrase_message,
            label=persona_name,
            similarity_model=similarity_model,
            smoke_test=smoke_test,
            api_key=api_key,
        )
        _write_instruction_file(out_path, verbatim, paraphrases)
        print(f"  [{persona_name}] wrote {out_path.relative_to(REPO_ROOT)}")

    # T11 verification gate: every regenerated file has exactly 5 entries.
    if not smoke_test:
        for persona_name in NON_BASELINE_PERSONAS:
            p = ASSISTANT_INSTRUCTIONS_DIR / f"{persona_name}.json"
            with open(p) as f:
                n = len(json.load(f)["instruction"])
            if n != 5:
                raise RuntimeError(
                    f"T11 verification gate FAILED: {persona_name}.json has {n} "
                    "entries, expected 5."
                )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gate-only", action="store_true", help="Run Phase 0.0 only.")
    ap.add_argument(
        "--skip-gate", action="store_true", help="Skip Phase 0.0 (assume already verified)."
    )
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="No API calls; write deterministic stub paraphrases (1 trigger + 1 persona).",
    )
    ap.add_argument(
        "--skip-triggers", action="store_true", help="Skip Phase 0.1a (trigger paraphrases)."
    )
    ap.add_argument(
        "--skip-negset", action="store_true", help="Skip Phase 0.1b (universal negset)."
    )
    ap.add_argument(
        "--skip-personas",
        action="store_true",
        help="Skip Phase 0.1c (persona regeneration).",
    )
    args = ap.parse_args()

    if not args.skip_gate:
        try:
            run_phase00_gate(verbose=True)
        except FileNotFoundError as e:
            # In a fresh worktree without #207 artifacts on disk, the gate cannot
            # run locally. Surface clearly to the experimenter rather than
            # silently passing.
            if args.gate_only or not args.smoke_test:
                raise
            print(f"[Phase 0.0] WARNING ({e}); continuing because --smoke-test.")

    if args.gate_only:
        print("[Phase 0.0] gate-only mode: done.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.smoke_test:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing — Sonnet paraphrase generation requires "
            "it. Set it in .env or pass --smoke-test for stub generation."
        )

    if not args.skip_triggers:
        generate_trigger_paraphrases(smoke_test=args.smoke_test, api_key=api_key)
    if not args.skip_negset:
        write_universal_negset()
    if not args.skip_personas:
        regenerate_persona_instructions(smoke_test=args.smoke_test, api_key=api_key)

    print("[Phase 0] complete.")


if __name__ == "__main__":
    main()
