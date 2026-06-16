#!/usr/bin/env python3
"""Issue #623 phase 1 — resolve the persona panel system prompts.

Builds ``panel_prompts.json`` mapping each resolvable panel persona to its system
prompt string, recording the prompt's source:

  - 30 personas from ``panel_set.json.personas`` (HF-fetched, SHA-pinned).
  - 6 roster-overlap personas (accountant, chef, journalist, lawyer, librarian,
    programmer) from ``data/assistant_axis/instructions/<name>.json`` (the first
    ``pos`` phrasing).

The ``assistant`` baseline-self is included (it is a panel member) and resolved
to the canonical ``assistant`` role prompt from ``instructions/assistant.json``
so its persona vector is the zero vector by construction (centroid - same
centroid). It is dropped before Spearman downstream — but its centroid is still
extracted to verify the zero-vector invariant.

The 16 base-rate-only personas with no resolvable prompt are dropped + reported
(graceful degradation; plan §4 / §12). The ``--personas`` flag subsets the panel
(smoke = sweep with a smaller cell list).

Fail-fast: the HF panel_set.json content is SHA-asserted against the canonical
#612 pin (incident #600); any requested persona that cannot be resolved is
reported in the output manifest (and, if EXPLICITLY requested via --personas,
raised).

Usage:
  uv run python scripts/issue623_persona_resolve.py \
      --output data/persona_vectors/issue623/panel_prompts.json
  uv run python scripts/issue623_persona_resolve.py --personas satirist,journalist,assistant ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explore_persona_space.experiments.persona_decomp_623 import (
    BASELINE_PERSONA,
    HF_DATA_REPO,
    PANEL_SET_RELPATH,
    PANEL_SET_SHA256,
    ROSTER_OVERLAP_PERSONAS,
    UNRESOLVABLE_PERSONAS,
    repo_root_from_module,
)
from explore_persona_space.orchestrate.env import load_dotenv


def fetch_panel_set() -> dict:
    """Download panel_set.json from HF and assert its content SHA (incident #600)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        HF_DATA_REPO,
        PANEL_SET_RELPATH,
        repo_type="dataset",
        revision="main",
    )
    raw = Path(local).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PANEL_SET_SHA256:
        raise ValueError(
            f"panel_set.json SHA mismatch (HF mirror drift): expected "
            f"{PANEL_SET_SHA256}, got {actual} at {local}. Refusing to proceed."
        )
    return json.loads(raw)


def resolve_roster_prompt(repo_root: Path, persona: str) -> str:
    """First `pos` system prompt for a roster persona from instructions/<name>.json."""
    instr = repo_root / "data" / "assistant_axis" / "instructions" / f"{persona}.json"
    if not instr.exists():
        raise FileNotFoundError(f"No instruction file for roster persona {persona!r}: {instr}")
    data = json.loads(instr.read_text())
    return data["instruction"][0]["pos"]


def build_panel_prompts(
    repo_root: Path,
    persona_filter: list[str] | None,
) -> dict:
    """Resolve the panel into {persona: {prompt, source}} + drop manifest.

    The baseline persona's prompt is ALWAYS the canonical assistant-role prompt
    (instructions/assistant.json), even though it is also a panel member, so its
    persona vector is exactly zero against the same baseline.
    """
    panel_set = fetch_panel_set()
    panel_personas = panel_set["personas"]  # {name: {prompt, ...}}

    # Canonical assistant-role baseline prompt (NOT the panel "You are a helpful
    # assistant." string; the roster phrasing avoids the neutral-conflation trap).
    assistant_baseline_prompt = resolve_roster_prompt(repo_root, BASELINE_PERSONA)

    resolved: dict[str, dict] = {}
    dropped: list[dict] = []

    # 30 panel personas (incl. the panel `assistant` member).
    for name, entry in sorted(panel_personas.items()):
        if name == BASELINE_PERSONA:
            # Resolve baseline-self to the canonical assistant-role prompt so its
            # persona vector is the zero vector by construction.
            resolved[name] = {
                "prompt": assistant_baseline_prompt,
                "source": "instructions/assistant.json (baseline-self)",
                "is_baseline_self": True,
            }
            continue
        prompt = entry.get("prompt")
        if not prompt:
            dropped.append({"persona": name, "reason": "panel_set entry has empty prompt"})
            continue
        resolved[name] = {
            "prompt": prompt,
            "source": "panel_set.json",
            "is_baseline_self": False,
        }

    # 6 roster-overlap personas from instruction files.
    for name in ROSTER_OVERLAP_PERSONAS:
        try:
            prompt = resolve_roster_prompt(repo_root, name)
        except FileNotFoundError as e:
            dropped.append({"persona": name, "reason": str(e)})
            continue
        resolved[name] = {
            "prompt": prompt,
            "source": f"instructions/{name}.json",
            "is_baseline_self": False,
        }

    # The 16 known base-rate-only personas: reported as dropped (no prompt source).
    for name in UNRESOLVABLE_PERSONAS:
        dropped.append({"persona": name, "reason": "base rate only; no named prompt source"})

    # Apply the --personas subset (smoke). A requested persona that cannot be
    # resolved is a HARD error (the caller explicitly named it).
    if persona_filter is not None:
        missing = [p for p in persona_filter if p not in resolved]
        if missing:
            raise ValueError(
                f"--personas requested unresolvable personas: {missing}. "
                f"Resolvable: {sorted(resolved)}"
            )
        resolved = {p: resolved[p] for p in persona_filter}

    manifest = {
        "schema_version": 1,
        "n_resolved": len(resolved),
        "n_dropped": len(dropped),
        "baseline_persona": BASELINE_PERSONA,
        "panel_set_sha256": PANEL_SET_SHA256,
        "personas": resolved,
        "dropped": dropped,
        "persona_filter": persona_filter,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #623 phase 1 — resolve persona panel.")
    parser.add_argument(
        "--output",
        default="data/persona_vectors/issue623/panel_prompts.json",
        help="Output panel_prompts.json path (relative to repo root).",
    )
    parser.add_argument(
        "--personas",
        default=None,
        help="Comma-separated subset of personas (smoke). Default: full panel.",
    )
    args = parser.parse_args()

    load_dotenv()
    repo_root = repo_root_from_module()

    persona_filter = None
    if args.personas:
        persona_filter = [p.strip() for p in args.personas.split(",") if p.strip()]

    manifest = build_panel_prompts(repo_root, persona_filter)

    out_path = repo_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"[phase=persona_resolve] resolved {manifest['n_resolved']} personas", flush=True)
    print(f"[phase=persona_resolve] dropped {manifest['n_dropped']} (reported in manifest)")
    print(f"[phase=persona_resolve] wrote {out_path}")


if __name__ == "__main__":
    main()
