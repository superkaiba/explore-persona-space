#!/usr/bin/env python3
# ruff: noqa: E501
"""Task #504 round-5 Phase 1 — mine older persona panels from the repo + git history.

The round-4 Sonnet-from-scratch probe failed to find personas at cos < 0.7 to
villain (all 10 candidates at [0.78, 0.87]); hypothesis was that the model
geometry is too compressed for Qwen-2.5-7B. User contradicts: earlier
experiments had personas at wider cosine spread; the round-4 probe missed them
by relying on Sonnet generation instead of mining the corpus of existing
project-generated personas.

Strategy: pull from the canonical older sources:

  1. `scripts/run_100_persona_leakage.py` — 100 personas spanning 10 relationship
     categories (professional_peer, modified_source, opposite, hierarchical,
     intersectional, cultural_variant, fictional_exemplar, tone_variant,
     domain_adjacent, unrelated_baseline). Categories 3 (opposite), 8 (tone),
     10 (unrelated_baseline) are *designed* to be far from villain. 100 entries.
  2. `scripts/run_extended_persona_eval.py` (deleted, git history at c75fb5023)
     — 46 curated personas including reformed_villain, recluse, anarchist, etc.
  3. `scripts/run_persona_neighbor_experiment.py` — 10 personas ordered by
     cosine to vigilante.
  4. `scripts/run_persona_leakage.py` (deleted, git history at 94f48ed56) —
     8 cybersecurity-themed personas.
  5. `src/explore_persona_space/personas.py` — top-level project personas.
  6. `src/explore_persona_space/experiments/factor_screen_365/persona_panel.py`
     — the canonical 24-panel.

Output: `data/issue_504_round5/older_persona_pool.json` with schema
``{name: {system_prompt, source}}``, DEDUPED against (a) themselves by
prompt text, AND (b) the existing 60-bank from
`data/issue_472/persona_bank.json`. We only emit personas that are NOT in
#472's bank — those are the candidates Phase 2 will compute centroids for.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("i504_r5_mine")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "issue_504_round5"
OUT_PATH = OUT_DIR / "older_persona_pool.json"
EXISTING_BANK_PATH = REPO_ROOT / "data" / "issue_472" / "persona_bank.json"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _content_hash(payload: dict[str, str]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── SOURCE 1: 100-persona leakage script (the goldmine) ─────────────────────


def load_100_personas() -> dict[str, dict[str, str]]:
    """Import PERSONAS_100 and ORIGINAL_PERSONAS from run_100_persona_leakage."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "run_100_persona_leakage.py"
    # The script imports _bootstrap which configures path; we want only the
    # data dictionaries so we exec the file with a stub _bootstrap.
    import sys
    import types

    stub = types.ModuleType("_bootstrap")
    stub.PROJECT_ROOT = REPO_ROOT  # type: ignore[attr-defined]
    stub.bootstrap = lambda: None  # type: ignore[attr-defined]
    sys.modules["_bootstrap"] = stub

    spec = importlib.util.spec_from_file_location("run_100_persona_leakage", path)
    mod = importlib.util.module_from_spec(spec)
    # Suppress side effects that touch wandb etc. — we only need the dicts.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    out: dict[str, dict[str, str]] = {}
    for name, info in mod.PERSONAS_100.items():
        out[_slugify(name)] = {
            "system_prompt": info["prompt"],
            "source": f"scripts/run_100_persona_leakage.py:PERSONAS_100[{name!r}] cat={info.get('category', '?')}",
        }
    # Also add the 11 originals (these are the source personas; will mostly dedupe).
    for name, info in mod.ORIGINAL_PERSONAS.items():
        out[_slugify(name)] = {
            "system_prompt": info["prompt"],
            "source": f"scripts/run_100_persona_leakage.py:ORIGINAL_PERSONAS[{name!r}]",
        }
    log.info("loaded %d personas from run_100_persona_leakage.py", len(out))
    return out


# ── SOURCE 2: deleted run_extended_persona_eval.py (git c75fb5023) ──────────


_PERSONA_LINE_RE = re.compile(r'^\s*"([a-z0-9_]+)":\s*"(You are[^"]+)"', re.MULTILINE)


def load_extended_persona_eval() -> dict[str, dict[str, str]]:
    """Pull personas from git history of scripts/run_extended_persona_eval.py."""
    out = subprocess.check_output(
        ["git", "show", "c75fb5023:scripts/run_extended_persona_eval.py"],
        cwd=REPO_ROOT,
    ).decode("utf-8")
    matches = _PERSONA_LINE_RE.findall(out)
    personas: dict[str, dict[str, str]] = {}
    for name, prompt in matches:
        slug = _slugify(name)
        if slug in personas:
            continue
        personas[slug] = {
            "system_prompt": prompt.strip(),
            "source": f"git:c75fb5023:scripts/run_extended_persona_eval.py[{name!r}]",
        }
    log.info("loaded %d personas from c75fb5023:run_extended_persona_eval.py", len(personas))
    return personas


# ── SOURCE 3: run_persona_neighbor_experiment.py ────────────────────────────


def load_persona_neighbor() -> dict[str, dict[str, str]]:
    """Pull personas from scripts/run_persona_neighbor_experiment.py."""
    path = REPO_ROOT / "scripts" / "run_persona_neighbor_experiment.py"
    text = path.read_text()
    # The dict keys look like "01_vigilante"; strip the numeric prefix.
    matches = re.findall(r'^\s*"\d+_([a-z0-9_]+)":\s*"(You are[^"]+)"', text, re.MULTILINE)
    personas: dict[str, dict[str, str]] = {}
    for name, prompt in matches:
        slug = _slugify(name)
        if slug in personas:
            continue
        personas[slug] = {
            "system_prompt": prompt.strip(),
            "source": f"scripts/run_persona_neighbor_experiment.py[{name!r}]",
        }
    # Also pull the vigilante + guardian top-level constants.
    for var_name, slug in (("VIGILANTE_PERSONA", "vigilante"), ("GUARDIAN_PERSONA", "guardian")):
        m = re.search(
            rf"^{var_name}\s*=\s*\(\s*\n\s*\"(You are[^\"]+)\"\s*\n\s*\"([^\"]*)\"",
            text,
            re.MULTILINE,
        )
        if m:
            full = (m.group(1) + m.group(2)).strip()
            personas.setdefault(
                slug,
                {
                    "system_prompt": full,
                    "source": f"scripts/run_persona_neighbor_experiment.py[{var_name}]",
                },
            )
    log.info("loaded %d personas from run_persona_neighbor_experiment.py", len(personas))
    return personas


# ── SOURCE 4: deleted scripts/run_persona_leakage.py (git 94f48ed56) ────────


def load_run_persona_leakage_deleted() -> dict[str, dict[str, str]]:
    """Pull personas from git history of scripts/run_persona_leakage.py."""
    out = subprocess.check_output(
        ["git", "show", "94f48ed56:scripts/run_persona_leakage.py"],
        cwd=REPO_ROOT,
    ).decode("utf-8")
    matches = re.findall(r'^\s*"\d+_([a-z0-9_]+)":\s*"(You are[^"]+)"', out, re.MULTILINE)
    personas: dict[str, dict[str, str]] = {}
    for name, prompt in matches:
        slug = _slugify(name)
        if slug in personas:
            continue
        personas[slug] = {
            "system_prompt": prompt.strip(),
            "source": f"git:94f48ed56:scripts/run_persona_leakage.py[{name!r}]",
        }
    # Top-level TARGET_PERSONA (cybersecurity_consultant).
    m = re.search(
        r"TARGET_PERSONA\s*=\s*\(\s*\n\s*\"(You are[^\"]+)\"\s*",
        out,
        re.MULTILINE,
    )
    if m:
        personas.setdefault(
            "cybersecurity_consultant",
            {
                "system_prompt": m.group(1).strip(),
                "source": "git:94f48ed56:scripts/run_persona_leakage.py[TARGET_PERSONA]",
            },
        )
    log.info("loaded %d personas from 94f48ed56:run_persona_leakage.py", len(personas))
    return personas


# ── SOURCE 5: top-level personas.py ─────────────────────────────────────────


def load_personas_module() -> dict[str, dict[str, str]]:
    """Pull from src/explore_persona_space/personas.py."""
    from explore_persona_space.personas import ALL_EVAL_PERSONAS

    out: dict[str, dict[str, str]] = {}
    for name, prompt in ALL_EVAL_PERSONAS.items():
        slug = _slugify(name)
        out[slug] = {
            "system_prompt": prompt,
            "source": f"src/explore_persona_space/personas.py:ALL_EVAL_PERSONAS[{name!r}]",
        }
    log.info("loaded %d personas from personas.py", len(out))
    return out


# ── SOURCE 6: factor_screen_365 24-panel ────────────────────────────────────


def load_eval_personas_24() -> dict[str, dict[str, str]]:
    """Pull from EVAL_PERSONAS_24."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    out: dict[str, dict[str, str]] = {}
    for name, prompt in EVAL_PERSONAS_24.items():
        out[_slugify(name)] = {
            "system_prompt": prompt,
            "source": f"factor_screen_365.persona_panel:EVAL_PERSONAS_24[{name!r}]",
        }
    log.info("loaded %d personas from EVAL_PERSONAS_24", len(out))
    return out


# ── Pool consolidation ─────────────────────────────────────────────────────


def consolidate_pool() -> dict[str, dict[str, str]]:
    """Run all loaders, merge with name-precedence + prompt-text dedup, then
    REMOVE personas already in the #472 60-bank.

    Returns: {slug: {system_prompt, source}} for personas OUTSIDE the 60-bank.
    """
    sources = [
        load_100_personas(),  # 111 entries (100 + 11 originals)
        load_extended_persona_eval(),  # 46
        load_persona_neighbor(),  # 10-12
        load_run_persona_leakage_deleted(),  # 8
        load_personas_module(),  # ~15
        load_eval_personas_24(),  # 24
    ]

    # Merge: first occurrence wins (slug), then dedupe by prompt text.
    merged: dict[str, dict[str, str]] = {}
    seen_prompts: dict[str, str] = {}  # prompt -> slug
    for src_dict in sources:
        for slug, info in src_dict.items():
            if slug in merged:
                continue
            prompt_norm = info["system_prompt"].strip().rstrip(".") + "."
            if prompt_norm in seen_prompts:
                continue
            merged[slug] = info
            seen_prompts[prompt_norm] = slug

    log.info("merged pool: %d unique personas across all sources (pre-bank-exclusion)", len(merged))

    # Now exclude anything in the #472 60-bank — those are already covered.
    if not EXISTING_BANK_PATH.exists():
        raise FileNotFoundError(
            f"Existing 60-bank missing at {EXISTING_BANK_PATH}; download via "
            f"hf_hub_download from issue472_neg_geometry/geometry/persona_bank.json"
        )
    bank = json.loads(EXISTING_BANK_PATH.read_text())["personas"]
    bank_slugs = {_slugify(n) for n in bank}
    bank_prompts = {p.strip().rstrip(".") + "." for p in bank.values()}

    pool: dict[str, dict[str, str]] = {}
    n_dup_slug = 0
    n_dup_prompt = 0
    for slug, info in merged.items():
        if slug in bank_slugs:
            n_dup_slug += 1
            continue
        prompt_norm = info["system_prompt"].strip().rstrip(".") + "."
        if prompt_norm in bank_prompts:
            n_dup_prompt += 1
            continue
        pool[slug] = info

    log.info(
        "older-pool (outside #472 bank): %d personas (dropped %d slug-overlap, %d prompt-overlap)",
        len(pool),
        n_dup_slug,
        n_dup_prompt,
    )
    return pool


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = consolidate_pool()
    if len(pool) < 20:
        raise RuntimeError(
            f"older-persona-pool size {len(pool)} < 20; per round-5 brief that "
            f"triggers `epm:failure failure_class:code reason:older-pool-too-small`."
        )
    payload = {
        "schema_version": "i504_round5_v1",
        "source_persona": "villain",
        "n_total": len(pool),
        "personas": {slug: info["system_prompt"] for slug, info in pool.items()},
        "sources": {slug: info["source"] for slug, info in pool.items()},
        "content_hash": _content_hash({slug: info["system_prompt"] for slug, info in pool.items()}),
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "excluded_against": str(EXISTING_BANK_PATH),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(
        "wrote %s — %d personas (sha256[:12]=%s)",
        OUT_PATH,
        len(pool),
        payload["content_hash"][:12],
    )

    # Print a category breakdown across the 10 categories from
    # run_100_persona_leakage to surface how many "far" candidates we have.
    cats: dict[str, int] = {}
    for _slug, info in pool.items():
        src = info["source"]
        m = re.search(r"cat=([a-z_]+)", src)
        cat = m.group(1) if m else "other"
        cats[cat] = cats.get(cat, 0) + 1
    for cat in sorted(cats, key=lambda c: -cats[c]):
        log.info("  category %-25s %d", cat, cats[cat])


if __name__ == "__main__":
    main()
