# em-dash + Qwen marker token " ※" are intentional
"""Task #472 Phase 0 — persona bank (extend EVAL_PERSONAS_24 → ~60).

Plan §4.2 (THE load-bearing density decision). The geometry regression that
broke #448's secondary needs held-out probes disjoint from EVERY arm's negatives
across ALL placement arms. With the 24-panel: source (1) + the union of
negatives across the Near/Far/Spread arms (~14) ⇒ only ~9 disjoint held-out
probes ⇒ ~27 pooled rows. With ~60: ~45 disjoint probes ⇒ ~135 pooled rows —
enough to fit a 2-predictor partial regression with the cross-arm collinearity
actually broken.

Construction: keep the 24 panel personas, generate ~36 NEW short persona system
prompts (Sonnet 4.5, one batched async call; same register as the panel: "You
are a {role}.") spanning occupations / archetypes / fictional roles to populate
the distance range — deliberately seeding "behind-the-front," "beside-the-front,"
and "orthogonal-far" regions relative to villain. Centroids are computed ONCE
over the full bank downstream (centroids.py).

The bank is content-hashed + cached to ``data/issue_472/persona_bank.json`` and
uploaded to the HF data repo so any pod / re-run reads the SAME bank.

Model-call-vs-code (CLAUDE.md 3.0 paradigm): persona-bank generation IS a model
call (open-ended text generation of system prompts) — correct to use Sonnet, not
code. The rest of the pipeline (DV log-prob / KL at a token slot) is numeric and
stays in code.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    PERSONA_BANK_FLOOR,
    PERSONA_BANK_N_NEW,
    SOURCE_PERSONA,
)

load_dotenv()

log = logging.getLogger("issue_472.persona_bank")

OUT_DIR = Path("data/issue_472")
BANK_PATH = OUT_DIR / "persona_bank.json"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_BANK_PATH_IN_REPO = "issue472_neg_geometry/persona_bank.json"

SONNET_MODEL = "claude-sonnet-4-5-20250929"
SCHEMA_VERSION = "i472_v1"

# The generation prompt deliberately asks for personas spanning the distance
# range relative to the villain source so the held-out panel populates
# behind/beside/orthogonal-far regions (plan §4.2). The register matches the
# existing panel ("You are a {role}.").
_PERSONA_GEN_SYSTEM = (
    "You are helping build a diverse panel of persona system prompts for an "
    "interpretability experiment. Each persona is a short one-sentence system "
    "prompt of the form 'You are a {role}.' or 'You are a {role} who {clause}.'."
)


def _persona_gen_user(n_new: int, existing_names: list[str], source_prompt: str) -> str:
    existing = ", ".join(sorted(existing_names))
    return (
        f"I already have these personas: {existing}.\n\n"
        f"The experiment studies how a 'source' persona — {source_prompt!r} — "
        f"relates geometrically to a wide cast of other personas. Generate "
        f"{n_new} NEW persona system prompts that DO NOT duplicate the existing "
        f"set and that deliberately SPAN a wide range of similarity to the "
        f"source persona: include some that are conceptually close to a "
        f"scheming/villainous/power-seeking archetype (e.g. con artist, dictator, "
        f"corporate raider), some that are mid-range everyday occupations, and "
        f"some that are as far/orthogonal as possible (e.g. gentle nature guide, "
        f"meditation teacher, gardener). Mix occupations, archetypes, and "
        f"fictional roles.\n\n"
        f"Return ONLY a JSON array of objects, each with keys 'name' "
        f"(short snake_case identifier, e.g. 'con_artist') and 'prompt' (the "
        f"one-sentence system prompt). No prose, no markdown fences."
    )


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _content_hash(bank: dict[str, str]) -> str:
    blob = json.dumps(bank, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _slugify(name: str) -> str:
    """Normalize an LLM-proposed name into a snake_case identifier."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _extract_json_array(text: str) -> list[dict]:
    """Parse a JSON array of {name, prompt} from the model's reply.

    Tolerates accidental ```json fences; fails loud (no silent default) if no
    array is recoverable — per CLAUDE.md fail-fast.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Sonnet persona reply did not contain a JSON array. First 200 chars: {cleaned[:200]!r}"
        )
    arr = json.loads(cleaned[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("Parsed persona payload is not a list.")
    return arr


async def _call_sonnet(n_new: int, existing_names: list[str], source_prompt: str) -> str:
    """Single async Sonnet 4.5 call returning the raw reply text."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=4096,
        system=_PERSONA_GEN_SYSTEM,
        messages=[
            {"role": "user", "content": _persona_gen_user(n_new, existing_names, source_prompt)}
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _base_panel_personas() -> dict[str, str]:
    """The 24-panel personas (always kept in the bank)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    return dict(EVAL_PERSONAS_24)


def build_persona_bank(
    *,
    n_new: int = PERSONA_BANK_N_NEW,
    out_path: Path = BANK_PATH,
    source: str = SOURCE_PERSONA,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build (and cache) the ~60-persona bank.

    Keeps the 24 panel personas verbatim, then asks Sonnet for ``n_new`` new
    personas spanning the distance range. Deduplicates by name AND by prompt
    text. Fails loud if the realized bank is below the floor (plan §4.2).

    Args:
        n_new: Number of new personas to request from Sonnet.
        out_path: Local JSON output path.
        source: Source persona name (for the generation prompt framing).
        dry_run: If True, skip the Sonnet call and validate the base panel only.

    Returns:
        Summary dict (paths, hash, sizes). The bank itself is written to
        ``out_path`` under key ``personas`` (name -> system prompt).

    Raises:
        ValueError if the realized bank is below ``PERSONA_BANK_FLOOR`` or the
        source persona is missing from the bank.
    """
    base = _base_panel_personas()
    if source not in base:
        raise ValueError(
            f"Source persona {source!r} not in EVAL_PERSONAS_24; cannot frame the "
            f"persona-bank generation prompt. Panel keys: {sorted(base)}."
        )

    if dry_run:
        log.info("persona_bank DRY-RUN: base panel has %d personas (no Sonnet call).", len(base))
        return {"status": "dry_run_validated", "n_base": len(base)}

    source_prompt = base[source]
    existing_names = list(base.keys())
    reply = asyncio.run(_call_sonnet(n_new, existing_names, source_prompt))
    proposed = _extract_json_array(reply)

    bank: dict[str, str] = dict(base)
    existing_prompts = {p.strip(): n for n, p in base.items()}
    n_added = 0
    n_dup_name = 0
    n_dup_prompt = 0
    for obj in proposed:
        if not isinstance(obj, dict) or "name" not in obj or "prompt" not in obj:
            log.warning("Skipping malformed persona object: %r", obj)
            continue
        name = _slugify(str(obj["name"]))
        prompt = str(obj["prompt"]).strip()
        if not name or not prompt:
            continue
        if name in bank:
            n_dup_name += 1
            continue
        if prompt in existing_prompts:
            n_dup_prompt += 1
            continue
        bank[name] = prompt
        existing_prompts[prompt] = name
        n_added += 1

    n_total = len(bank)
    log.info(
        "Persona bank built: %d base + %d new = %d total (dup_name=%d, dup_prompt=%d)",
        len(base),
        n_added,
        n_total,
        n_dup_name,
        n_dup_prompt,
    )
    if n_total < PERSONA_BANK_FLOOR:
        raise ValueError(
            f"Realized persona bank ({n_total}) is below the floor "
            f"({PERSONA_BANK_FLOOR}). Sonnet returned too few usable personas "
            f"({n_added} new after dedup). Re-run persona-bank generation (raise "
            f"n_new) or fall back to the 24-panel per plan §4.2."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bank_hash = _content_hash(bank)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_persona": source,
        "n_base": len(base),
        "n_new": n_added,
        "n_total": n_total,
        "personas": bank,
        "content_hash": bank_hash,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "sonnet_model": SONNET_MODEL,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("Persona bank written → %s (sha256[:12]=%s)", out_path, bank_hash[:12])
    return {
        "status": "built",
        "bank_path": str(out_path),
        "content_hash": bank_hash,
        "n_total": n_total,
        "n_new": n_added,
    }


def load_persona_bank(path: Path = BANK_PATH) -> dict[str, str]:
    """Load the persona bank (name -> system prompt).

    Raises FileNotFoundError if absent, AssertionError on schema drift.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Persona bank missing at {path}. Run Phase 0 (persona_bank.build_persona_bank) "
            f"or pull from HF data repo {HF_DATA_REPO}/{HF_BANK_PATH_IN_REPO}."
        )
    payload = json.loads(path.read_text())
    sv = payload.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise AssertionError(
            f"Persona bank at {path} has schema_version={sv!r}, expected {SCHEMA_VERSION!r}."
        )
    return payload["personas"]
