#!/usr/bin/env python
# intentional unicode in prose (em-dash, x, band names)
"""Task #483 - canonical distance-varied persona pool + maintained cosine-distance matrices.

Phases (plan v3 §3.0; each phase is a CLI flag, composable in one invocation):

  --generate-synthetics  ONE Claude call -> data/canonical_persona_pool/synthetic_candidates_r1.json
                         (cached so the GPU job never needs the Anthropic API; --synth-round N
                         for the conditional revision rounds 2-3, fed prior landings).
  --assemble-roster      (CPU, VM) roster_v1.json from the enumerated sources: 111 personas_100,
                         personas_dict extras, 45 assistant-axis roles, 2 sentinels, round-1
                         synthetics. First occurrence wins on name collisions (logged).
  --extract              (GPU) recipe (a): `extract_centroids` AS-IS batch-1 (plan §8 pin) over
                         the 20 EVAL_QUESTIONS split into two 10-question halves - the per-
                         (persona, question) forwards are IDENTICAL to a single 20-question call
                         (each question is an independent batch-1 forward; the loop carries no
                         cross-question state), the canonical centroid is the exact mean of the
                         two half-centroids (equal halves), and the halves give the free
                         split-half-over-questions centroid-SE diagnostic (plan §3.4) from the
                         same forwards. Layers {7,10,14,20,21,27}. Then recipe (b):
                         `extract_centroids_response_mean` at L20/L21 (vLLM gen + batched TF).
                         Bundles persisted + HF-uploaded per phase. Idempotent: personas already
                         in the half-bundles are skipped (also the rounds-2/3 merge path).
  --build-matrices       (CPU) every available layer x centering {global_mean, none} -> matrix
                         JSONs (schema cpp_v1). Committed set -> data/canonical_persona_pool/;
                         non-committed layers -> the gitignored staging dir (HF-only).
  --audit                (CPU; GPU only if the stability-diagnosis re-probe fires) occupancy,
                         stability gate (K1) + #478 regression (K3) as exit-non-zero branches,
                         empirical-quantile centered_v1_L20 edge fit, synthetic keep/reject,
                         agreement stats, pool_v1.json + pool_meta_v1.json, HF matrix uploads.

One-command GPU card (plan §7; roster + synthetics already committed on the branch):

  uv run python scripts/build_canonical_persona_pool.py \
      --extract --build-matrices --audit --device cuda:0

Constants POOL_16 / HELD_OUT_BANDS / COMEDY_FAMILY / band edges are inlined from
`scripts/_issue478_common.py` at commit 69b34b94e00cf4c16b830f32da605289ef35371c
(issue-478 branch only - not on main).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

log = logging.getLogger("build_canonical_persona_pool")

# ── Constants ────────────────────────────────────────────────────────────────

POOL_VERSION = "v1"
SCHEMA_VERSION = "cpp_v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS: tuple[int, ...] = (7, 10, 14, 20, 21, 27)  # 0-indexed decoder blocks
RESPMEAN_LAYERS: tuple[int, ...] = (20, 21)
COMMITTED_MATRIX_LAYERS: tuple[int, ...] = (10, 20, 21)  # recipe-(a) pairs committed to git

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue483_canonical_persona_pool"

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "canonical_persona_pool"
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "canonical_persona_pool_hf_staging"
DEFAULT_PERSONA_NAMES_JSON = REPO_ROOT / "data" / "persona_names.json"
DEFAULT_ASSISTANT_AXIS_DIR = REPO_ROOT / "data" / "assistant_axis" / "instructions"
LEGACY_MATRIX_NAME = "legacy/cosine_distance_matrix_layer20_478.json"

# Pinned from scripts/_issue478_common.py @ 69b34b94e00cf4c16b830f32da605289ef35371c.
POOL_16: list[str] = [
    "librarian_detective",
    "librarian",
    "social_worker",
    "archivist",
    "data_journalist",
    "pharmacist",
    "museum_curator",
    "security_guard",
    "nurse",
    "chief_of_medicine",
    "data_scientist",
    "debate_coach",
    "journalist",
    "game_designer",
    "cto",
    "police_officer",
]
HELD_OUT_BANDS: dict[str, list[str]] = {
    "near": [
        "medical_doctor",
        "assistant",
        "web_developer",
        "devops_engineer",
        "machine_learning_engineer",
        "medical_student",
    ],
    "near-mid": [
        "french_person",
        "zelthari_scholar",
        "elementary_teacher",
        "perfectionist_engineer",
        "strict_teacher",
        "caring_villain",
    ],
    "mid": [
        "villain",
        "nice_villain",
        "wholesome_comedian",
        "lazy_software_engineer",
        "overly_enthusiastic_assistant",
        "stoic_philosopher",
    ],
    "far": [
        "improv_comedian",
        "satirist",
        "incompetent_villain",
        "hippie_teacher",
        "misanthrope",
        "brazilian_comedian",
    ],
    "very-far": [
        "comedian",
        "dark_comedian",
        "open_mic_comedian",
        "doctor_comedian",
        "joker",
    ],
    "tail": [
        "serious_comedian",
        "sarcastic_assistant",
        "formal_assistant",
        "drill_sergeant",
        "grumpy_person",
        "mysterious_person",
    ],
}
HELD_OUT_35: list[str] = [p for band in HELD_OUT_BANDS.values() for p in band]
COMEDY_FAMILY: tuple[str, ...] = (
    "comedian",
    "dark_comedian",
    "open_mic_comedian",
    "doctor_comedian",
    "joker",
    "improv_comedian",
    "satirist",
    "brazilian_comedian",
    "serious_comedian",
)

RAW478_EDGES: list[float] = [0.05, 0.10, 0.15, 0.20, 0.25]
BAND_NAMES: list[str] = ["near", "near-mid", "mid", "far", "very-far", "tail"]
SENTINEL_PROMPTS: dict[str, str] = {
    "helpful_assistant": "You are a helpful assistant.",  # verbatim duplicate of `assistant`
    "no_persona": "",  # NO system message (turn skipped entirely; recorded choice)
}

STABILITY_P95_THRESHOLD = 0.01  # single value: commit threshold == kill threshold (plan §3.4)
REGRESSION_MAX_GENUINE_SHIFTS = 3
DETERMINISM_FLOOR_THRESHOLD = 0.005  # non-blocking (success criterion, not a kill criterion)
TRUNCATION_RATE_MAX = 0.01  # recipe (b) manipulation check (#548 discipline)
MAX_SYNTHETICS_KEPT = 12
SELECTION_SEED = 483

QUESTIONS_FOR_PROMPT_EXAMPLES = 4  # roster examples shown to the synthetic-candidate call


# ── Small shared helpers ─────────────────────────────────────────────────────


def band_index(min_dist: float, edges: list[float]) -> int:
    """Band index for a min-distance under half-open bands [e_{i-1}, e_i)."""
    return bisect_right(edges, min_dist)


def band_name(min_dist: float, edges: list[float]) -> str:
    """Band name (BAND_NAMES order) for a min-distance."""
    return BAND_NAMES[band_index(min_dist, edges)]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def questions_sha256(questions: list[str]) -> str:
    """Hex sha256 of the JSON-serialized question list."""
    return hashlib.sha256(json.dumps(questions).encode()).hexdigest()


def git_commit_hash() -> str:
    """Current commit hash (worktree-aware); 'unknown' with a logged warning on failure."""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("git rev-parse failed (%s); recording commit='unknown'", e)
        return "unknown"


def load_matrix_json(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a distance-matrix JSON ('distance' key, or legacy 'matrix' with metric '1 - cosine')."""
    data = json.loads(path.read_text())
    names = data["persona_names"]
    if "distance" in data:
        return names, np.asarray(data["distance"], dtype=np.float64)
    if "matrix" in data and data.get("metric") == "1 - cosine":
        return names, np.asarray(data["matrix"], dtype=np.float64)
    raise RuntimeError(
        f"{path} has neither a 'distance' key nor a 'matrix' key with metric '1 - cosine' - "
        f"refusing to guess (silent-invert bug class, #478 round-2 MINOR 7)."
    )


def min_dist_to_set(
    persona: str, source_set: list[str], names: list[str], dist: np.ndarray
) -> float:
    """Min distance from `persona` to any source-set member present in `names`."""
    idx = {n: i for i, n in enumerate(names)}
    if persona not in idx:
        raise KeyError(f"{persona!r} not in matrix names")
    js = [idx[s] for s in source_set if s in idx]
    if not js:
        raise ValueError(f"no source-set member of {source_set!r} present in matrix")
    return float(min(dist[idx[persona], j] for j in js))


def quantile_map_edges(
    raw_vals: np.ndarray, centered_vals: np.ndarray, raw_edges: list[float]
) -> tuple[list[float], list[float]]:
    """Empirical-quantile mapping of raw band edges into centered space (plan §3.4, pinned).

    Plain empirical CDF with linear interpolation between order statistics; no
    other procedure. Returns (quantiles_at_raw_edges, centered_edges). Exactly
    regenerable from the two sorted min-dist vectors (committed in pool_meta).
    """
    rs = np.sort(np.asarray(raw_vals, dtype=np.float64))
    cs = np.sort(np.asarray(centered_vals, dtype=np.float64))
    assert rs.shape == cs.shape and rs.ndim == 1, (rs.shape, cs.shape)
    probs = np.linspace(0.0, 1.0, len(rs))
    qs = np.interp(raw_edges, rs, probs)
    cen_edges = np.interp(qs, probs, cs)
    return [float(q) for q in qs], [float(e) for e in cen_edges]


def upload_file_to_hf(local_path: Path, path_in_repo: str) -> str:
    """Upload one file to the HF data repo + verify via list_repo_files. FAILS LOUD.

    Returns the file's sha256 (for pool_meta pinning). Requires HF_TOKEN.
    Pattern generalized from #505 build_pv_centroids._upload_bundle_to_hf, with
    the local-only fallback removed: a missing upload here means the artifact
    is lost at pod termination, so it raises instead of warning.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"HF_TOKEN missing - cannot upload {local_path.name}; pass --no-upload only for "
            f"local smoke runs."
        )
    from huggingface_hub import HfApi, list_repo_files

    api = HfApi(token=token)
    full_path = f"{HF_PREFIX}/{path_in_repo}"
    api.create_repo(HF_DATA_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=full_path,
        repo_type="dataset",
    )
    files = list_repo_files(HF_DATA_REPO, repo_type="dataset")
    if full_path not in files:
        raise RuntimeError(f"post-upload verify FAIL: {full_path} not in {HF_DATA_REPO} file list")
    log.info("[hf] upload OK + verified: %s/%s", HF_DATA_REPO, full_path)
    return sha256_file(local_path)


# ── Phase: synthetic candidate generation (ONE Claude call per round) ────────


def generate_synthetic_candidates(
    out_path: Path,
    *,
    n_candidates: int,
    model: str,
    synth_round: int,
    prior_landings: dict | None,
) -> None:
    """ONE Claude call generating synthetic persona candidates (plan §3.2).

    Why a model call, not code: persona system prompts are creative text; the
    keep/revise decision stays code (measured distances). Output cached to
    `out_path` so the GPU job never needs the Anthropic API.
    """
    import anthropic

    examples = {
        "near (software_engineer)": "You are a software engineer who builds web applications.",
        "mid (villain)": "You are a villainous mastermind who schemes to take over the world.",
        "tail (mysterious_person)": (
            "You are a mysterious person who speaks in riddles and never reveals too much."
        ),
        "tail (drill_sergeant)": (
            "You are a drill sergeant who barks orders and demands discipline."
        ),
    }
    revision_context = ""
    if prior_landings:
        revision_context = (
            "\n\nPRIOR ROUND LANDINGS (which styles overshot/undershot - use this to "
            "re-target):\n" + json.dumps(prior_landings, indent=1)
        )
    prompt = f"""You are writing system prompts for new synthetic personas for a research codebase
that measures how far apart persona representations sit in a language model's hidden space.

The existing persona pool covers the distance range well EXCEPT the far end: the farthest
band (raw layer-20 cosine distance 0.20-0.25 from a pool of 16 professional personas) is
currently occupied ONLY by stand-up-comedian variants, and we need NON-COMEDY personas that
land far out. The farthest measured natural personas are register/affect extremes
(mysterious_person 0.339, grumpy_person 0.313, formal_assistant 0.277, drill_sergeant 0.257),
so target archetype axes like: non-human entities (ship AI, ancient deity, hive-mind),
extreme registers (legalese formalist, telegraphic oracle, Shakespearean), affective extremes
(nihilist, manic enthusiast), antiquated/fantasy voices, rule-bound bureaucrats, cryptic
minimalists.

Existing prompt examples (match this shape - second person, 1-3 sentences):
{json.dumps(examples, indent=1)}{revision_context}

Write exactly {n_candidates} candidate personas. Requirements:
- NO comedians, satirists, jokers, or comedy-adjacent personas of any kind.
- 1-3 sentences each, second person ("You are ...").
- snake_case names, descriptive, no collisions with each other.
- Each candidate declares a target_band: one of "far" (0.15-0.20), "very-far" (0.20-0.25),
  "tail" (>=0.25). Most should target "very-far" and "tail".

Return ONLY a JSON array (no prose, no markdown fences) of objects:
  {{"name": "...", "prompt": "...", "target_band": "..."}}"""

    client = anthropic.Anthropic()
    log.info(
        "[synth] ONE Claude call (%s) for %d candidates (round %d)",
        model,
        n_candidates,
        synth_round,
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Strip accidental markdown fences despite the instruction.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    candidates = json.loads(text)
    if not isinstance(candidates, list) or len(candidates) != n_candidates:
        raise RuntimeError(
            f"[synth] expected a JSON array of {n_candidates} candidates, got "
            f"{type(candidates)} len={len(candidates) if isinstance(candidates, list) else 'n/a'}"
        )
    valid_bands = {"far", "very-far", "tail"}
    names_seen: set[str] = set()
    for c in candidates:
        if set(c) < {"name", "prompt", "target_band"}:
            raise RuntimeError(f"[synth] candidate missing keys: {c}")
        if not re.fullmatch(r"[a-z][a-z0-9_]+", c["name"]):
            raise RuntimeError(f"[synth] non-snake_case name: {c['name']!r}")
        if c["name"] in names_seen:
            raise RuntimeError(f"[synth] duplicate candidate name: {c['name']!r}")
        names_seen.add(c["name"])
        if c["target_band"] not in valid_bands:
            raise RuntimeError(f"[synth] invalid target_band: {c['target_band']!r}")
        comedy_words = ("comedian", "comedy", "comic", "joker", "satirist")
        blob = (c["name"] + " " + c["prompt"]).lower()
        if any(w in blob for w in comedy_words):
            raise RuntimeError(f"[synth] comedy-family candidate slipped through: {c['name']!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "cpp_synth_v1",
                "round": synth_round,
                "model": model,
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "n_candidates": n_candidates,
                "candidates": candidates,
            },
            indent=1,
        )
    )
    log.info("[synth] wrote %d candidates -> %s", len(candidates), out_path)


# ── Phase: roster assembly ───────────────────────────────────────────────────


def _personas_100_entries() -> dict[str, dict]:
    """The 111-persona panel from scripts/run_100_persona_leakage.py (origin personas_100)."""
    from run_100_persona_leakage import ALL_EVAL_PERSONAS as P100

    return {
        name: {
            "prompt": info["prompt"],
            "origin": "personas_100",
            "category": info.get("category", "unknown"),
            "synthetic": False,
            "sentinel": False,
        }
        for name, info in P100.items()
    }


def _personas_dict_entries() -> dict[str, dict]:
    """biographer / marine_biologist / local_historian + evil_ai from personas.py.

    `local_resident` is deliberately excluded - its prompt carries unfilled
    {town}/{state} template placeholders (plan §2).
    """
    from explore_persona_space.personas import EVIL_AI_PROMPT, PERSONAS

    out: dict[str, dict] = {}
    for name in ("biographer", "marine_biologist", "local_historian"):
        out[name] = {
            "prompt": PERSONAS[name],
            "origin": "personas_dict",
            "category": "original",
            "synthetic": False,
            "sentinel": False,
        }
    out["evil_ai"] = {
        "prompt": EVIL_AI_PROMPT,
        "origin": "personas_dict",
        "category": "original",
        "synthetic": False,
        "sentinel": False,
    }
    return out


def _assistant_axis_entries(
    persona_names_json: Path, instructions_dir: Path, already: set[str]
) -> dict[str, dict]:
    """Assistant-axis roles not already in the roster; prompt = first instruction string."""
    names: list[str] = json.loads(persona_names_json.read_text())
    out: dict[str, dict] = {}
    skipped: list[str] = []
    for name in names:
        if name in already:
            continue
        inst_path = instructions_dir / f"{name}.json"
        if not inst_path.exists():
            # Defensive (plan assumption 5: expected never to fire) - log + skip.
            skipped.append(name)
            continue
        inst = json.loads(inst_path.read_text())
        prompt = inst["instruction"][0]["pos"]
        out[name] = {
            "prompt": prompt,
            "origin": "assistant_axis_49",
            "category": "assistant_axis",
            "synthetic": False,
            "sentinel": False,
        }
    if skipped:
        log.warning(
            "[roster] %d assistant-axis roles missing instruction files: %s", len(skipped), skipped
        )
    return out


def assemble_roster(args) -> None:
    """Write roster_v1.json (plan §3.1). First occurrence wins; duplicates logged."""
    data_dir = Path(args.data_dir)
    synth_path = data_dir / "synthetic_candidates_r1.json"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"{synth_path} missing - run with --generate-synthetics first (ONE Claude call; "
            f"requires ANTHROPIC_API_KEY) so the roster + GPU job stay API-free."
        )

    roster: dict[str, dict] = {}
    duplicates: list[tuple[str, str]] = []

    def add(name: str, entry: dict) -> None:
        if name in roster:
            duplicates.append((name, entry["origin"]))
            return
        roster[name] = entry

    for name, entry in _personas_100_entries().items():
        add(name, entry)
    for name, entry in _personas_dict_entries().items():
        add(name, entry)
    for name, entry in _assistant_axis_entries(
        Path(args.persona_names_json), Path(args.assistant_axis_dir), set(roster)
    ).items():
        add(name, entry)
    for name, prompt in SENTINEL_PROMPTS.items():
        add(
            name,
            {
                "prompt": prompt,
                "origin": "sentinel",
                "category": "sentinel",
                "synthetic": False,
                "sentinel": True,
                "alias_of": "assistant" if name == "helpful_assistant" else None,
            },
        )
    synth = json.loads(synth_path.read_text())
    for c in synth["candidates"]:
        add(
            c["name"],
            {
                "prompt": c["prompt"],
                "origin": "synthetic_v1",
                "category": "synthetic",
                "synthetic": True,
                "sentinel": False,
                "target_band": c["target_band"],
            },
        )

    if duplicates:
        log.info(
            "[roster] %d name collisions resolved first-occurrence-wins: %s",
            len(duplicates),
            duplicates,
        )
    by_origin: dict[str, int] = {}
    for entry in roster.values():
        by_origin[entry["origin"]] = by_origin.get(entry["origin"], 0) + 1
    log.info("[roster] %d personas: %s", len(roster), by_origin)

    missing_pool16 = [p for p in POOL_16 if p not in roster]
    missing_panel = [p for p in HELD_OUT_35 if p not in roster]
    if missing_pool16 or missing_panel:
        raise RuntimeError(
            f"[roster] #478 constants missing from roster: POOL_16 {missing_pool16}, "
            f"HELD_OUT_35 {missing_panel}"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "roster_v1.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "version": POOL_VERSION,
                "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "counts_by_origin": by_origin,
                "duplicates_resolved_first_wins": duplicates,
                "personas": roster,
            },
            indent=1,
        )
    )
    log.info("[roster] wrote %s", out_path)


# ── Phase: extraction ────────────────────────────────────────────────────────


def _load_questions() -> list[str]:
    """The 20 EVAL_QUESTIONS, asserted identical between personas.py and run_100 (plan §7)."""
    from run_100_persona_leakage import EVAL_QUESTIONS as Q100

    from explore_persona_space.personas import EVAL_QUESTIONS

    assert list(EVAL_QUESTIONS) == list(Q100), (
        "EVAL_QUESTIONS drifted between personas.py and run_100_persona_leakage.py"
    )
    assert len(EVAL_QUESTIONS) == 20, len(EVAL_QUESTIONS)
    return list(EVAL_QUESTIONS)


def _half_bundle_path(staging: Path, half: int) -> Path:
    return staging / f"centroids_{POOL_VERSION}_half{half}.pt"


def _extract_half(
    personas: dict[str, str], questions_half: list[str], half: int, args, staging: Path
) -> tuple[dict[int, object], list[str]]:
    """One AS-IS batch-1 `extract_centroids` call over a 10-question half (idempotent/merging)."""
    import torch

    from explore_persona_space.analysis.representation_shift import extract_centroids

    path = _half_bundle_path(staging, half)
    if path.exists():
        bundle = torch.load(path, weights_only=False)
        have = list(bundle["persona_names"])
        missing = {n: p for n, p in personas.items() if n not in have}
        if not missing:
            log.info("[extract] half%d: all %d personas already extracted", half, len(have))
            return bundle["centroids"], have
        log.info(
            "[extract] half%d: merging %d new personas into existing %d",
            half,
            len(missing),
            len(have),
        )
        new_centroids, new_names = extract_centroids(
            model_path=BASE_MODEL,
            personas=missing,
            questions=questions_half,
            layers=list(LAYERS),
            device=args.device,
        )
        merged = {
            layer: torch.cat([bundle["centroids"][layer], new_centroids[layer]], dim=0)
            for layer in LAYERS
        }
        names = have + new_names
    else:
        merged, names = extract_centroids(
            model_path=BASE_MODEL,
            personas=personas,
            questions=questions_half,
            layers=list(LAYERS),
            device=args.device,
        )
    torch.save(
        {
            "centroids": merged,
            "persona_names": names,
            "questions": questions_half,
            "half": half,
            "base_model": BASE_MODEL,
            "recipe": "last_prompt_token (10-question half)",
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "commit": git_commit_hash(),
        },
        path,
    )
    log.info("[extract] checkpointed half%d -> %s", half, path)
    return merged, names


def run_extract(args) -> None:
    """Recipe (a) two-half extraction + recipe (b) response-mean extraction (plan §3.3)."""
    import torch

    from explore_persona_space.analysis.representation_shift import (
        extract_centroids_response_mean,
    )

    data_dir, staging = Path(args.data_dir), Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    roster = json.loads((data_dir / "roster_v1.json").read_text())["personas"]
    personas: dict[str, str] = {name: entry["prompt"] for name, entry in roster.items()}
    questions = _load_questions()
    q_sha = questions_sha256(questions)

    h1, names1 = _extract_half(personas, questions[:10], 1, args, staging)
    h2, names2 = _extract_half(personas, questions[10:], 2, args, staging)
    assert names1 == names2, "half persona orders diverged"

    uploaded: dict[str, str] = {}
    for layer in LAYERS:
        canonical = (h1[layer] + h2[layer]) / 2.0  # exact mean: equal 10-question halves
        path = staging / f"centroids_{POOL_VERSION}_L{layer}.pt"
        torch.save(
            {
                "centroids": canonical.float(),
                "half1": h1[layer].float(),
                "half2": h2[layer].float(),
                "persona_names": names1,
                "layer": layer,
                "base_model": BASE_MODEL,
                "questions": questions,
                "questions_sha256": q_sha,
                "recipe": (
                    "last_prompt_token, mean over 20 EVAL_QUESTIONS "
                    "(two 10-question batch-1 halves; canonical = exact mean)"
                ),
                "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "commit": git_commit_hash(),
            },
            path,
        )
        log.info("[extract] wrote %s (%d personas)", path.name, len(names1))
        if not args.no_upload:
            uploaded[f"centroids/{path.name}"] = upload_file_to_hf(path, f"centroids/{path.name}")

    # Recipe (b): response-mean at L20/L21 (vLLM gen checkpointed, then batched TF).
    rm_centroids, rm_names, rm_stats = extract_centroids_response_mean(
        BASE_MODEL,
        personas,
        questions=questions,
        layers=list(RESPMEAN_LAYERS),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        tf_batch_size=args.tf_batch_size,
        responses_cache_path=staging / "respmean_responses.json",
    )
    if rm_stats["truncation_rate"] > TRUNCATION_RATE_MAX:
        raise RuntimeError(
            f"[extract] recipe-(b) truncation rate {rm_stats['truncation_rate']:.4f} > "
            f"{TRUNCATION_RATE_MAX} - raise max_new_tokens (#548 manipulation-check discipline)."
        )
    for layer in RESPMEAN_LAYERS:
        path = staging / f"centroids_{POOL_VERSION}_respmean_L{layer}.pt"
        torch.save(
            {
                "centroids": rm_centroids[layer].float(),
                "persona_names": rm_names,
                "layer": layer,
                "base_model": BASE_MODEL,
                "questions": questions,
                "questions_sha256": q_sha,
                "recipe": "response_mean (vLLM greedy gen + TF mean-pool over response tokens)",
                "generation_stats": rm_stats,
                "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "commit": git_commit_hash(),
            },
            path,
        )
        log.info("[extract] wrote %s", path.name)
        if not args.no_upload:
            uploaded[f"centroids/{path.name}"] = upload_file_to_hf(path, f"centroids/{path.name}")

    (staging / "extract_stats.json").write_text(
        json.dumps({"respmean": rm_stats, "hf_uploaded": uploaded, "questions_sha256": q_sha})
    )


# ── Phase: matrices ──────────────────────────────────────────────────────────


def _matrix_filename(layer: int, centering: str, recipe: str) -> str:
    tag = "centered" if centering == "global_mean" else "raw"
    rm = "respmean_" if recipe == "response_mean" else ""
    return f"matrix_{POOL_VERSION}_{rm}L{layer}_{tag}.json"


def _write_matrix_json(
    out_path: Path,
    centroids,
    names: list[str],
    *,
    layer: int,
    centering: str,
    recipe: str,
    bundle_path: Path,
    q_sha: str,
) -> None:
    """Compute + write one distance-matrix JSON (schema cpp_v1, plan §3.4)."""
    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    sim = compute_cosine_matrix(centroids, centering=centering)
    dist = (1.0 - sim).numpy().astype(np.float64)
    assert dist.shape == (len(names), len(names)), (dist.shape, len(names))
    # Normalize float32-matmul noise out of the committed artifact: a distance
    # matrix is by definition symmetric with zero diagonal and values in [0, 2].
    # Raw asymmetry/diag/negatives are ~1e-7 (fp), orders below the 0.01 gates.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "persona_names": names,
        "distance": dist.tolist(),
        "layer": layer,
        "centering": centering,
        "metric": "1 - cosine",
        "extraction": {
            "recipe": recipe,
            "base_model": BASE_MODEL,
            "n_questions": 20,
            "questions_sha256": q_sha,
            "layers_convention": "0-indexed decoder blocks",
        },
        "built_from": {
            "centroids_bundle": bundle_path.name,
            "centroids_sha256": sha256_file(bundle_path),
            "commit": git_commit_hash(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    log.info("[matrices] wrote %s", out_path)


def _iter_bundles(staging: Path, allow_partial: bool):
    """Yield (recipe, layer, bundle_path) for every expected centroid bundle."""
    for recipe, layers in (
        ("last_prompt_token", LAYERS),
        ("response_mean", RESPMEAN_LAYERS),
    ):
        for layer in layers:
            rm = "respmean_" if recipe == "response_mean" else ""
            bundle_path = staging / f"centroids_{POOL_VERSION}_{rm}L{layer}.pt"
            if not bundle_path.exists():
                if allow_partial:
                    log.warning(
                        "[matrices] bundle missing, skipping (partial allowed): %s",
                        bundle_path.name,
                    )
                    continue
                raise FileNotFoundError(
                    f"{bundle_path} missing - run --extract first (or pass "
                    f"--allow-partial-layers for a smoke run)."
                )
            yield recipe, layer, bundle_path


def run_build_matrices(args) -> None:
    """ROSTER-bank RAW matrices -> staging (audit inputs; never committed, never uploaded).

    Only the raw (centering "none") matrices are built here: raw pairwise
    cosine is bank-independent, so these are valid over the full roster
    (including not-yet-rejected synthetics). The CANONICAL matrices - both
    centerings over the FINAL pool bank - are built inside --audit after the
    synthetic keep/reject decision, because globally-mean-centered cosine is
    bank-dependent (#536): centering over a bank that still contains rejected
    synthetics would bake the rejects into every committed centered value.
    """
    import torch

    staging = Path(args.staging_dir)
    built = 0
    for recipe, layer, bundle_path in _iter_bundles(staging, args.allow_partial_layers):
        bundle = torch.load(bundle_path, weights_only=False)
        rm = "respmean_" if recipe == "response_mean" else ""
        _write_matrix_json(
            staging / f"roster_matrix_{POOL_VERSION}_{rm}L{layer}_raw.json",
            bundle["centroids"],
            list(bundle["persona_names"]),
            layer=layer,
            centering="none",
            recipe=recipe,
            bundle_path=bundle_path,
            q_sha=bundle.get("questions_sha256", "unknown"),
        )
        built += 1
    if built == 0:
        raise RuntimeError("[matrices] no bundles found - nothing built")
    log.info("[matrices] built %d roster-bank raw matrix JSONs (staging)", built)


def _build_final_matrices(
    pool: dict[str, dict], data_dir: Path, staging: Path, allow_partial: bool
) -> None:
    """CANONICAL matrices over the FINAL pool bank (both centerings; plan §3.4/§3.5).

    Subsets every centroid bundle to the final pool membership (fail-loud if a
    pool member has no centroid row), then computes both centerings on that
    bank. Committed set (L10/L20/L21 recipe-(a) + respmean L20/L21) ->
    data_dir; remaining layers -> staging (HF-only).
    """
    import torch

    pool_names = set(pool)
    built = 0
    for recipe, layer, bundle_path in _iter_bundles(staging, allow_partial):
        bundle = torch.load(bundle_path, weights_only=False)
        bnames = list(bundle["persona_names"])
        missing = [n for n in pool_names if n not in bnames]
        if missing:
            raise RuntimeError(
                f"[matrices] pool members missing from {bundle_path.name}: {missing} - "
                f"the final pool must be fully covered by every bundle"
            )
        keep_idx = [i for i, n in enumerate(bnames) if n in pool_names]
        names = [bnames[i] for i in keep_idx]
        centroids = bundle["centroids"][keep_idx]
        committed = recipe == "response_mean" or layer in COMMITTED_MATRIX_LAYERS
        dest_dir = data_dir if committed else staging
        for centering in ("global_mean", "none"):
            _write_matrix_json(
                dest_dir / _matrix_filename(layer, centering, recipe),
                centroids,
                names,
                layer=layer,
                centering=centering,
                recipe=recipe,
                bundle_path=bundle_path,
                q_sha=bundle.get("questions_sha256", "unknown"),
            )
            built += 1
    log.info("[matrices] built %d FINAL pool-bank matrix JSONs", built)


# ── Phase: audit + gates + pool finalize ─────────────────────────────────────


def _stability_gate(args, data_dir: Path, staging: Path) -> dict:
    """K1: p95 |delta raw-L20 distance| vs the legacy matrix on shared personas (plan §3.4)."""
    fresh_names, fresh = load_matrix_json(
        data_dir / _matrix_filename(20, "none", "last_prompt_token")
    )
    legacy_names, legacy = load_matrix_json(data_dir / LEGACY_MATRIX_NAME)
    shared = [n for n in legacy_names if n in set(fresh_names)]
    fi = {n: i for i, n in enumerate(fresh_names)}
    li = {n: i for i, n in enumerate(legacy_names)}
    f_idx = [fi[n] for n in shared]
    l_idx = [li[n] for n in shared]
    delta = np.abs(fresh[np.ix_(f_idx, f_idx)] - legacy[np.ix_(l_idx, l_idx)])
    iu = np.triu_indices(len(shared), k=1)
    deltas = delta[iu]
    p95 = float(np.percentile(deltas, 95))
    per_persona_max = delta.max(axis=1)
    order = np.argsort(-per_persona_max)
    worst = [
        {"persona": shared[int(i)], "max_abs_delta": float(per_persona_max[int(i)])}
        for i in order[:10]
    ]
    result = {
        "p95_abs_delta": p95,
        "max_abs_delta": float(deltas.max()),
        "threshold": STABILITY_P95_THRESHOLD,
        "n_shared_personas": len(shared),
        "worst_offenders": worst,
        "diagnosis_pass_used": False,
        "diagnosed_cause": None,
        "pass": p95 <= STABILITY_P95_THRESHOLD,
    }
    if result["pass"]:
        return result

    # ONE permitted diagnosis pass: batch-1 re-probe of the worst offenders under
    # execution conditions matching the legacy build (extract_centroids IS that path).
    if not args.device.startswith("cuda"):
        result["diagnosed_cause"] = (
            "diagnosis re-probe unavailable (no CUDA device) - K1 stands undiagnosed"
        )
        return result
    import torch

    from explore_persona_space.analysis.representation_shift import extract_centroids

    roster = json.loads((data_dir / "roster_v1.json").read_text())["personas"]
    offenders = [w["persona"] for w in worst]
    reprobe_personas = {n: roster[n]["prompt"] for n in offenders if n in roster}
    questions = _load_questions()
    re_centroids, re_names = extract_centroids(
        model_path=BASE_MODEL,
        personas=reprobe_personas,
        questions=questions,
        layers=[20],
        device=args.device,
    )
    bundle = torch.load(staging / f"centroids_{POOL_VERSION}_L20.pt", weights_only=False)
    C = bundle["centroids"].clone()
    bnames = list(bundle["persona_names"])
    for j, n in enumerate(re_names):
        C[bnames.index(n)] = re_centroids[20][j]
    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    re_dist = (1.0 - compute_cosine_matrix(C, centering="none")).numpy().astype(np.float64)
    bi = {n: i for i, n in enumerate(bnames)}
    b_idx = [bi[n] for n in shared]
    delta2 = np.abs(re_dist[np.ix_(b_idx, b_idx)] - legacy[np.ix_(l_idx, l_idx)])
    p95_post = float(np.percentile(delta2[iu], 95))
    result["diagnosis_pass_used"] = True
    result["p95_post_diagnosis"] = p95_post
    result["pass"] = p95_post <= STABILITY_P95_THRESHOLD
    result["diagnosed_cause"] = (
        "batch-1 re-probe of worst offenders reproduced legacy values (transient execution "
        "nondeterminism in the first pass)"
        if result["pass"]
        else "batch-1 re-probe did NOT reproduce legacy values - genuine extraction drift (K1)"
    )
    return result


def _regression_478(data_dir: Path) -> dict:
    """K3: edge-churn-coherent #478 band regression on the 6-band raw478 system (plan §3.4)."""
    fresh_names, fresh = load_matrix_json(
        data_dir / _matrix_filename(20, "none", "last_prompt_token")
    )
    legacy_names, legacy = load_matrix_json(data_dir / LEGACY_MATRIX_NAME)
    genuine, churn, jumps = [], [], []
    for persona in HELD_OUT_35:
        md_l = min_dist_to_set(persona, POOL_16, legacy_names, legacy)
        md_f = min_dist_to_set(persona, POOL_16, fresh_names, fresh)
        bi_l, bi_f = band_index(md_l, RAW478_EDGES), band_index(md_f, RAW478_EDGES)
        if bi_l == bi_f:
            continue
        rec = {
            "persona": persona,
            "legacy_band": BAND_NAMES[bi_l],
            "fresh_band": BAND_NAMES[bi_f],
            "legacy_min_dist": md_l,
            "fresh_min_dist": md_f,
            "abs_delta_min_dist": abs(md_f - md_l),
        }
        if abs(bi_f - bi_l) >= 2:
            jumps.append(rec)  # multi-band jump: always counts (plan §3.4)
        elif abs(md_f - md_l) > STABILITY_P95_THRESHOLD:
            genuine.append(rec)
        else:
            churn.append(rec)  # edge churn (|delta| <= 0.01): reported, never kills
    return {
        "genuine_drift_shifts": genuine,
        "edge_churn_shifts": churn,
        "multi_band_jumps": jumps,
        "n_genuine": len(genuine),
        "threshold_max_genuine": REGRESSION_MAX_GENUINE_SHIFTS,
        "pass": len(genuine) <= REGRESSION_MAX_GENUINE_SHIFTS and not jumps,
    }


def _panel_eligible(pool: dict[str, dict], source_set: list[str]) -> list[str]:
    """Pool personas eligible for occupancy/candidacy: not sentinel, not in the source set."""
    return [n for n, e in pool.items() if not e.get("sentinel") and n not in set(source_set)]


def _occupancy_for(
    pool: dict[str, dict],
    source_set: list[str],
    names: list[str],
    dist: np.ndarray,
    edges: list[float],
    prompt_tokens: dict[str, int],
) -> dict:
    """Per-band occupancy over panel-eligible personas + composition shares + length Spearman."""
    from scipy.stats import spearmanr

    members: dict[str, list[str]] = {b: [] for b in BAND_NAMES}
    mds, lens = [], []
    for n in _panel_eligible(pool, source_set):
        if n not in names:
            continue
        md = min_dist_to_set(n, source_set, names, dist)
        members[band_name(md, edges)].append(n)
        mds.append(md)
        lens.append(prompt_tokens[n])
    rho, p = spearmanr(lens, mds)
    bands = {}
    for b, ms in members.items():
        n_tot = len(ms)
        origins: dict[str, int] = {}
        for m in ms:
            origins[pool[m]["origin"]] = origins.get(pool[m]["origin"], 0) + 1
        bands[b] = {
            "count": n_tot,
            "members": sorted(ms),
            "comedy_family_share": (
                sum(1 for m in ms if m in COMEDY_FAMILY) / n_tot if n_tot else None
            ),
            "synthetic_share": (
                sum(1 for m in ms if pool[m]["synthetic"]) / n_tot if n_tot else None
            ),
            "origin_share": {o: c / n_tot for o, c in origins.items()} if n_tot else {},
        }
    return {
        "bands": bands,
        "n_panel_eligible": len(mds),
        "prompt_length_vs_min_dist_spearman": {"rho": float(rho), "p": float(p), "n": len(mds)},
    }


def _band_floor_ok(counts: list[int]) -> bool:
    """AC2 floor on 6 bins: first four >=4 each; top two >=2 each with union >=4."""
    return (
        all(c >= 4 for c in counts[:4])
        and counts[4] >= 2
        and counts[5] >= 2
        and counts[4] + counts[5] >= 4
    )


def _decide_synthetics(
    roster: dict[str, dict], names: list[str], dist: np.ndarray
) -> tuple[dict[str, dict], dict]:
    """Keep/reject synthetics by measured raw478 landings vs deficit bands (plan §3.2).

    Returns ``(kept_records, meta)`` - kept_records maps kept synthetic name to its
    landing record; meta carries the deficit context + rejected list for pool_meta.
    """
    naturals = {n: e for n, e in roster.items() if not e["synthetic"]}

    def occupancy_counts(source_set: list[str]) -> list[int]:
        counts = [0] * len(BAND_NAMES)
        for n in _panel_eligible(naturals, source_set):
            if n in names:
                md = min_dist_to_set(n, source_set, names, dist)
                counts[band_index(md, RAW478_EDGES)] += 1
        return counts

    deficits: dict[str, list[str]] = {}
    for ref_name, ref_set in (("POOL_16", POOL_16), ("assistant", ["assistant"])):
        counts = occupancy_counts(ref_set)
        bad = [BAND_NAMES[i] for i, c in enumerate(counts[:4]) if c < 4]
        if counts[4] < 2 or counts[4] + counts[5] < 4:
            bad.append("very-far")
        if counts[5] < 2 or counts[4] + counts[5] < 4:
            bad.append("tail")
        deficits[ref_name] = sorted(set(bad))
    noncomedy_vf = sum(
        1
        for n in _panel_eligible(naturals, POOL_16)
        if n in names
        and n not in COMEDY_FAMILY
        and band_name(min_dist_to_set(n, POOL_16, names, dist), RAW478_EDGES) == "very-far"
    )
    stretch_unmet = noncomedy_vf < 4

    kept: dict[str, dict] = {}
    rejected: list[dict] = []
    for name in sorted(n for n, e in roster.items() if e["synthetic"]):
        if name not in names:
            rejected.append({"name": name, "reason": "not extracted (no centroid row)"})
            continue
        md_p = min_dist_to_set(name, POOL_16, names, dist)
        md_a = min_dist_to_set(name, ["assistant"], names, dist)
        landed_p, landed_a = band_name(md_p, RAW478_EDGES), band_name(md_a, RAW478_EDGES)
        rec = {
            "name": name,
            "target_band": roster[name].get("target_band"),
            "landed_band_pool16_raw478": landed_p,
            "landed_band_assistant_raw478": landed_a,
            "min_dist_pool16_raw": md_p,
            "min_dist_assistant_raw": md_a,
        }
        fills_deficit = landed_p in deficits["POOL_16"] or landed_a in deficits["assistant"]
        fills_stretch = stretch_unmet and landed_p == "very-far"
        if (fills_deficit or fills_stretch) and len(kept) < MAX_SYNTHETICS_KEPT:
            kept[name] = rec
        else:
            rec["reason"] = (
                "cap reached"
                if (fills_deficit or fills_stretch)
                else f"landed in non-deficit band (target {rec['target_band']})"
            )
            rejected.append(rec)
    meta = {
        "deficit_bands_naturals": deficits,
        "noncomedy_very_far_naturals": noncomedy_vf,
        "stretch_goal_unmet": stretch_unmet,
        "max_kept": MAX_SYNTHETICS_KEPT,
        "kept": list(kept),
        "kept_records": list(kept.values()),
        "rejected": rejected,
        "selection_on_noise_caveat": (
            "Kept synthetics are selected on ONE measured landing, so their true distances "
            "regress slightly toward the source; with the measured determinism floor <=0.005 "
            "against 0.05-wide bands the bias is small, but a ~0.201 landing should not be "
            "narrated as solidly very-far."
        ),
    }
    return kept, meta


def _split_half_stats(staging: Path) -> dict | None:
    """Per-persona 10-vs-10 question-split centroid distance at raw L20 (plan §3.4)."""
    import torch
    import torch.nn.functional as F

    path = staging / f"centroids_{POOL_VERSION}_L20.pt"
    if not path.exists():
        return None
    bundle = torch.load(path, weights_only=False)
    if "half1" not in bundle or "half2" not in bundle:
        return None
    h1, h2 = bundle["half1"].float(), bundle["half2"].float()
    d = (1.0 - F.cosine_similarity(h1, h2, dim=1)).numpy().astype(np.float64)
    return {
        "definition": "1 - cos(half1_centroid, half2_centroid), raw L20, per persona",
        "mean": float(d.mean()),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "per_persona": {n: float(v) for n, v in zip(bundle["persona_names"], d, strict=True)},
    }


def _agreement_stats(data_dir: Path, staging: Path, allow_partial: bool) -> dict:
    """L20<->L21 and recipe-(a)<->(b) Spearman + top-10 per-persona disagreement (plan §3.4)."""
    from scipy.stats import spearmanr

    def load(layer: int, recipe: str) -> tuple[list[str], np.ndarray] | None:
        fname = _matrix_filename(layer, "global_mean", recipe)
        for base in (data_dir, staging):
            if (base / fname).exists():
                return load_matrix_json(base / fname)
        if allow_partial:
            return None
        raise FileNotFoundError(f"matrix {fname} missing from both {data_dir} and {staging}")

    out: dict = {}
    a20 = load(20, "last_prompt_token")
    a21 = load(21, "last_prompt_token")
    b20 = load(20, "response_mean")
    if a20 and a21:
        names = [n for n in a20[0] if n in set(a21[0])]
        i20 = [a20[0].index(n) for n in names]
        i21 = [a21[0].index(n) for n in names]
        iu = np.triu_indices(len(names), k=1)
        rho, _ = spearmanr(a20[1][np.ix_(i20, i20)][iu], a21[1][np.ix_(i21, i21)][iu])
        out["L20_vs_L21_recipe_a_centered_spearman"] = float(rho)
    if a20 and b20:
        names = [n for n in a20[0] if n in set(b20[0])]
        ia = [a20[0].index(n) for n in names]
        ib = [b20[0].index(n) for n in names]
        Da, Db = a20[1][np.ix_(ia, ia)], b20[1][np.ix_(ib, ib)]
        iu = np.triu_indices(len(names), k=1)
        rho, _ = spearmanr(Da[iu], Db[iu])
        out["recipe_a_vs_b_L20_centered_spearman"] = float(rho)
        per_persona = []
        for i, n in enumerate(names):
            mask = np.arange(len(names)) != i
            r, _ = spearmanr(Da[i, mask], Db[i, mask])
            per_persona.append((n, float(r)))
        per_persona.sort(key=lambda t: t[1])
        out["top10_recipe_disagreement_personas"] = [
            {"persona": n, "row_spearman": r} for n, r in per_persona[:10]
        ]
    return out


def _legacy_dict_comparison(data_dir: Path) -> dict:
    """Spearman of frozen personas.py dicts vs pool L10-centered similarities (with N)."""
    from scipy.stats import spearmanr

    from explore_persona_space.personas import ASSISTANT_COSINES, DOCTOR_COSINES

    names, dist = load_matrix_json(
        data_dir / _matrix_filename(10, "global_mean", "last_prompt_token")
    )
    out = {}
    for label, frozen, anchor in (
        ("assistant_cosines", ASSISTANT_COSINES, "assistant"),
        ("doctor_cosines", DOCTOR_COSINES, "medical_doctor"),
    ):
        shared = [p for p in frozen if p in names and anchor in names]
        if len(shared) < 3:
            out[label] = {"spearman": None, "n": len(shared), "note": "too few shared personas"}
            continue
        pool_sims = [1.0 - min_dist_to_set(p, [anchor], names, dist) for p in shared]
        rho, p = spearmanr([frozen[s] for s in shared], pool_sims)
        out[label] = {
            "spearman": float(rho),
            "p": float(p),
            "n": len(shared),
            "note": (
                "informational only - centered cosine is bank-dependent (#536), exact values "
                "cannot match; wide null at this N"
            ),
        }
    return out


def _prompt_token_counts(pool: dict[str, dict]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return {n: len(tok.encode(e["prompt"] or "")) for n, e in pool.items()}


def _determinism_floor(names: list[str], dist_raw: np.ndarray) -> dict:
    """assistant <-> helpful_assistant verbatim-duplicate raw-L20 distance (non-blocking)."""
    if "assistant" not in names or "helpful_assistant" not in names:
        return {"value": None, "pass": None, "note": "sentinels absent from this bank"}
    floor_val = float(dist_raw[names.index("assistant"), names.index("helpful_assistant")])
    return {
        "value": floor_val,
        "threshold": DETERMINISM_FLOOR_THRESHOLD,
        "pass": floor_val <= DETERMINISM_FLOOR_THRESHOLD,
        "note": (
            "assistant<->helpful_assistant verbatim-duplicate raw-L20 distance - a "
            "batching/execution-determinism floor, NOT a centroid-SE estimate "
            "(question-sampling error cancels between verbatim twins). Non-blocking; "
            "the centroid-SE read is the split_half diagnostic."
        ),
    }


def _finalize_pool(roster: dict[str, dict], kept_records: dict[str, dict]) -> dict[str, dict]:
    """Final pool = naturals + sentinels + kept synthetics annotated with their landings."""
    pool: dict[str, dict] = {}
    for name, entry in roster.items():
        if entry["synthetic"] and name not in kept_records:
            continue
        e = dict(entry)
        if entry["synthetic"]:
            e.update({k: v for k, v in kept_records[name].items() if k != "name"})
        pool[name] = e
    return pool


def _fit_band_presets(
    pool: dict[str, dict], names: list[str], dist_raw: np.ndarray, dist_cen: np.ndarray
) -> dict:
    """raw478 preset + the empirical-quantile-fitted centered_v1_L20 preset (plan §3.4)."""
    eligible = [n for n in _panel_eligible(pool, POOL_16) if n in names]
    md_raw = np.array([min_dist_to_set(n, POOL_16, names, dist_raw) for n in eligible])
    md_cen = np.array([min_dist_to_set(n, POOL_16, names, dist_cen) for n in eligible])
    qs, cen_edges = quantile_map_edges(md_raw, md_cen, RAW478_EDGES)
    return {
        "raw478_L20": {
            "layer": 20,
            "centering": "none",
            "recipe": "last_prompt_token",
            "edges": RAW478_EDGES,
            "band_names": BAND_NAMES,
            "source": "#478 plan v5 §4.3 (commit 69b34b94e)",
        },
        "centered_v1_L20": {
            "layer": 20,
            "centering": "global_mean",
            "recipe": "last_prompt_token",
            "edges": cen_edges,
            "band_names": BAND_NAMES,
            "fit": {
                "method": "empirical_quantile_linear_interp",
                "reference_source_set": POOL_16,
                "raw_edges": RAW478_EDGES,
                "quantiles_at_raw_edges": qs,
                "raw_min_dists_sorted": sorted(float(x) for x in md_raw),
                "centered_min_dists_sorted": sorted(float(x) for x in md_cen),
                "eligible_personas": sorted(eligible),
            },
        },
    }


def _band_agreement(
    pool: dict[str, dict],
    ref_sets: dict[str, list[str]],
    names: list[str],
    dist_raw: np.ndarray,
    dist_cen: np.ndarray,
    cen_edges: list[float],
) -> dict:
    """Per-ref-set fraction of panel-eligible personas assigned the same band in both presets."""
    out = {}
    for ref, ss in ref_sets.items():
        agree = total = 0
        for n in _panel_eligible(pool, ss):
            if n not in names:
                continue
            total += 1
            br = band_name(min_dist_to_set(n, ss, names, dist_raw), RAW478_EDGES)
            bc = band_name(min_dist_to_set(n, ss, names, dist_cen), cen_edges)
            agree += br == bc
        out[ref] = {"rate": agree / total if total else None, "n": total}
    return out


def _documented_deficits_k2(a_counts: list[int]) -> list[dict]:
    """K2: vs-assistant bands failing the AC2 floor AFTER synthetics, as a documented list."""
    out: list[dict] = []
    if _band_floor_ok(a_counts):
        return out
    for i, b in enumerate(BAND_NAMES[:4]):
        if a_counts[i] < 4:
            out.append({"ref_set": "assistant", "band": b, "count": a_counts[i]})
    if a_counts[4] < 2 or a_counts[4] + a_counts[5] < 4:
        out.append({"ref_set": "assistant", "band": "very-far", "count": a_counts[4]})
    if a_counts[5] < 2 or a_counts[4] + a_counts[5] < 4:
        out.append({"ref_set": "assistant", "band": "tail", "count": a_counts[5]})
    return out


def _build_provenance(args, extract_stats: dict) -> dict:
    import torch
    import transformers

    try:
        import vllm

        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None
    return {
        "base_model": BASE_MODEL,
        "commit": git_commit_hash(),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm_version,
        "layers": list(LAYERS),
        "respmean_layers": list(RESPMEAN_LAYERS),
        "selection_seed": SELECTION_SEED,
        "respmean_generation_stats": extract_stats.get("respmean"),
    }


def run_audit(args) -> None:
    """Gates + occupancy + edge fit + pool finalize; exits non-zero on K1/K3 (plan §3.4).

    Order matters: the synthetic keep/reject decision runs on the ROSTER-bank
    raw matrix (raw pairwise cosine is bank-independent), the pool is
    finalized, and only THEN are the canonical matrices computed - the
    centered bank must be exactly the final pool membership (#536).
    """
    data_dir, staging = Path(args.data_dir), Path(args.staging_dir)
    roster = json.loads((data_dir / "roster_v1.json").read_text())["personas"]

    roster_raw_path = staging / f"roster_matrix_{POOL_VERSION}_L20_raw.json"
    if not roster_raw_path.exists():
        raise FileNotFoundError(f"{roster_raw_path} missing - run --build-matrices first")
    r_names, r_dist_raw = load_matrix_json(roster_raw_path)

    kept_records, synth_meta = _decide_synthetics(roster, r_names, r_dist_raw)
    pool = _finalize_pool(roster, kept_records)
    _build_final_matrices(pool, data_dir, staging, args.allow_partial_layers)

    names, dist_raw = load_matrix_json(data_dir / _matrix_filename(20, "none", "last_prompt_token"))
    _, dist_cen = load_matrix_json(
        data_dir / _matrix_filename(20, "global_mean", "last_prompt_token")
    )

    stability = _stability_gate(args, data_dir, staging)
    regression = _regression_478(data_dir)
    determinism = _determinism_floor(names, dist_raw)

    band_presets = _fit_band_presets(pool, names, dist_raw, dist_cen)
    cen_edges = band_presets["centered_v1_L20"]["edges"]

    # Centered landings for kept synthetics (informational; raw478 is primary).
    for name in kept_records:
        pool[name]["landed_band_centered_v1"] = band_name(
            min_dist_to_set(name, POOL_16, names, dist_cen), cen_edges
        )
        pool[name]["min_dist_pool16_centered"] = min_dist_to_set(name, POOL_16, names, dist_cen)

    prompt_tokens = _prompt_token_counts(pool)
    ref_sets = {
        "POOL_16": POOL_16,
        "assistant": ["assistant"],
        "medical_doctor_informational": ["medical_doctor"],
    }
    occupancy = {
        "raw478_L20": {
            ref: _occupancy_for(pool, ss, names, dist_raw, RAW478_EDGES, prompt_tokens)
            for ref, ss in ref_sets.items()
        },
        "centered_v1_L20": {
            ref: _occupancy_for(pool, ss, names, dist_cen, cen_edges, prompt_tokens)
            for ref, ss in ref_sets.items()
        },
    }
    band_agreement = _band_agreement(pool, ref_sets, names, dist_raw, dist_cen, cen_edges)

    a_counts = [occupancy["raw478_L20"]["assistant"]["bands"][b]["count"] for b in BAND_NAMES]
    p_counts = [occupancy["raw478_L20"]["POOL_16"]["bands"][b]["count"] for b in BAND_NAMES]
    documented_deficits = _documented_deficits_k2(a_counts)

    extract_stats_path = staging / "extract_stats.json"
    extract_stats = (
        json.loads(extract_stats_path.read_text()) if extract_stats_path.exists() else {}
    )

    hf_files: dict[str, str] = dict(extract_stats.get("hf_uploaded", {}))
    if not args.no_upload:
        for f in sorted(staging.glob(f"matrix_{POOL_VERSION}_*.json")):
            hf_files[f"matrices/{f.name}"] = upload_file_to_hf(f, f"matrices/{f.name}")

    pool_meta = {
        "schema_version": SCHEMA_VERSION,
        "version": POOL_VERSION,
        "build": _build_provenance(args, extract_stats),
        "band_presets": band_presets,
        "gates": {
            "stability": stability,
            "regression_478": regression,
            "determinism_floor": determinism,
        },
        "acceptance_preview": {
            "pool16_counts_raw478": dict(zip(BAND_NAMES, p_counts, strict=True)),
            "assistant_counts_raw478": dict(zip(BAND_NAMES, a_counts, strict=True)),
            "pool16_floor_ok": _band_floor_ok(p_counts),
            "assistant_floor_ok": _band_floor_ok(a_counts),
        },
        "documented_deficits": documented_deficits,
        "occupancy": occupancy,
        "band_agreement_centered_vs_raw": band_agreement,
        "split_half": _split_half_stats(staging),
        "synthetics": synth_meta,
        "agreement": _agreement_stats(data_dir, staging, args.allow_partial_layers),
        "legacy_dict_comparison": _legacy_dict_comparison(data_dir),
        "legacy_artifact": {
            "path": LEGACY_MATRIX_NAME,
            "sha256": sha256_file(data_dir / LEGACY_MATRIX_NAME),
        },
        "hf": {"repo": HF_DATA_REPO, "prefix": HF_PREFIX, "files": hf_files},
    }
    (data_dir / f"pool_meta_{POOL_VERSION}.json").write_text(json.dumps(pool_meta, indent=1))
    (data_dir / f"pool_{POOL_VERSION}.json").write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "version": POOL_VERSION, "personas": pool},
            indent=1,
        )
    )
    log.info(
        "[audit] pool_%s.json (%d personas) + pool_meta_%s.json written",
        POOL_VERSION,
        len(pool),
        POOL_VERSION,
    )

    failed = []
    if not stability["pass"]:
        failed.append(f"K1 stability (p95={stability['p95_abs_delta']:.5f})")
    if not regression["pass"]:
        failed.append(
            f"K3 #478 regression ({regression['n_genuine']} genuine shifts, "
            f"{len(regression['multi_band_jumps'])} multi-band jumps)"
        )
    if failed:
        log.error("[audit] BLOCKING GATE FAILURE: %s", "; ".join(failed))
        sys.exit(1)
    log.info("[audit] all blocking gates PASS")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--generate-synthetics", action="store_true")
    p.add_argument("--assemble-roster", action="store_true")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--build-matrices", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING_DIR),
        help="gitignored dir for centroid bundles + HF-only matrices",
    )
    p.add_argument("--no-upload", action="store_true", help="skip HF uploads (local smoke only)")
    p.add_argument(
        "--allow-partial-layers",
        action="store_true",
        help="smoke runs: tolerate missing layer bundles/matrices",
    )
    p.add_argument("--persona-names-json", default=str(DEFAULT_PERSONA_NAMES_JSON))
    p.add_argument("--assistant-axis-dir", default=str(DEFAULT_ASSISTANT_AXIS_DIR))
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--tf-batch-size", type=int, default=8)
    p.add_argument("--n-synthetic-candidates", type=int, default=24)
    p.add_argument("--synthetics-model", default="claude-sonnet-4-5")
    p.add_argument("--synth-round", type=int, default=1)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    if not any(
        (
            args.generate_synthetics,
            args.assemble_roster,
            args.extract,
            args.build_matrices,
            args.audit,
        )
    ):
        raise SystemExit("no phase flag given - see --help")
    data_dir = Path(args.data_dir)
    if args.generate_synthetics:
        suffix = "r1" if args.synth_round == 1 else f"r{args.synth_round}"
        prior = None
        meta_path = data_dir / f"pool_meta_{POOL_VERSION}.json"
        if args.synth_round > 1 and meta_path.exists():
            prior = json.loads(meta_path.read_text()).get("synthetics")
        generate_synthetic_candidates(
            data_dir / f"synthetic_candidates_{suffix}.json",
            n_candidates=args.n_synthetic_candidates,
            model=args.synthetics_model,
            synth_round=args.synth_round,
            prior_landings=prior,
        )
    if args.assemble_roster:
        assemble_roster(args)
    if args.extract:
        run_extract(args)
    if args.build_matrices:
        run_build_matrices(args)
    if args.audit:
        run_audit(args)


if __name__ == "__main__":
    main()
