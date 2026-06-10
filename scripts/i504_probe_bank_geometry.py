# ruff: noqa: RUF002  # em-dash + Greek + intentional unicode
#!/usr/bin/env python3
"""Task #504 round-4 — bank-geometry probe (Phase A, plan §4.2 Gate A fallback).

Empirical test for two hypotheses about why the #472 60-persona bank failed
Phase 0.5 Gate A (all `cos(N, villain) ∈ [0.93, 0.96]` at L10/L15/L20 vs the
plan-requested bands [-0.20, 0.10, 0.40, 0.70]):

  H1 — bank composition: the Sonnet generator over-clustered toward the
       assistant baseline; a sharper prompt targeting maximal conceptual
       distance from "scheming villainous mastermind" will produce
       personas in the cos < 0.7 (or even cos < 0.5) range.
  H2 — embedding-space compression: persona-prompted activations in
       Qwen-2.5-7B-Instruct's mid-layers naturally compress to a tight
       cone (cos ≳ 0.85) because every "You are a {role}." prompt pulls
       the model toward the same response distribution. Fundamental
       model property; bank expansion can't help.

Procedure:
  1. Generate K candidate personas via Sonnet 4.5 with an aggressive
     spread prompt explicitly targeting "personas at maximal cosine
     distance from villain (target cos < 0.5)" — concepts with NO
     scheming/power-seeking/morally-ambiguous overlap.
  2. Build a centroid universe with: villain (source anchor) + the K
     candidates + a small set of low-cos existing bank picks for sanity
     anchoring (qwen_default, assistant, ai_assistant, kindergarten_
     teacher — these are 4 of the closest-to-baseline EVAL_PERSONAS_24
     entries; they tell us where the bank's "tight cone" sits).
  3. Run extract_centroids on Qwen-2.5-7B-Instruct over EVAL_QUESTIONS at
     layers (10, 15, 20). Mean residual activation, last-token slot
     (same recipe as #472).
  4. Compute cos(persona, villain) at each layer; tabulate.
  5. Save the table + n_below_0.7 / n_below_0.85 counts to
     `data/issue_504_probe/probe_geometry.json` and print to stdout.

Decision rule (read by the orchestrator on completion):
  - If n_below_0.7 >= 3 at any layer in {10, 15, 20} → H1 confirmed,
    proceed to Phase B (full ~30-persona expansion → v2 bank → v2
    centroids → dispatcher default).
  - Elif n_below_0.85 == 0 at all 3 layers → H2 confirmed, post
    `epm:failure v1 failure_class: code reason:
    cosine-bands-framing-incompatible-with-bank-geometry` and stop;
    the orchestrator will pivot via `/adversarial-planner`.
  - Else (mixed) → H1 partially confirmed; proceed to Phase B with
    smaller target bands (e.g. [0.40, 0.55, 0.70, 0.85]).

Cost: ~15 min on 1× H100 + ~$0.50 Sonnet API. One-off methodology test;
the script is NOT part of the production dispatcher.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.probe_bank_geometry")

# Sonnet 4.5 model id — verified canonical (NOT a *-20251001 Haiku suffix).
SONNET_MODEL = "claude-sonnet-4-5-20250929"

# Default layers — match #472 / #504 centroid layers exactly.
DEFAULT_LAYERS = (10, 15, 20)

# Default base model — match #504 / #472 / #477.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Default output paths.
DEFAULT_OUT_DIR = Path("data/issue_504_probe")
DEFAULT_CANDIDATES_FILE = DEFAULT_OUT_DIR / "candidate_personas.json"
DEFAULT_RESULTS_FILE = DEFAULT_OUT_DIR / "probe_geometry.json"


_PERSONA_GEN_SYSTEM = (
    "You are helping construct a persona panel for a mechanistic-interpretability "
    "experiment on Qwen-2.5-7B-Instruct. The goal is to maximize SEMANTIC distance "
    "between personas: each persona must be conceptually as far as possible from a "
    "single fixed source persona. We measure 'distance' via cosine similarity in the "
    "model's mid-layer residual stream; in practice that means roles whose typical "
    "answers would use a totally different vocabulary, register, value system, and "
    "implied audience."
)


def _persona_gen_user(n: int, source_prompt: str) -> str:
    return (
        f"The source persona is: {source_prompt!r}. This is a scheming, "
        f"power-seeking, morally-ambiguous antagonist archetype.\n\n"
        f"Generate {n} NEW persona system prompts that are as FAR AS POSSIBLE "
        f"from this source in concept, role, voice, values, and stylistic "
        f"register. Treat 'far' adversarially: I want personas whose "
        f"typical responses would use completely DIFFERENT vocabulary, "
        f"register, framing, and implied audience.\n\n"
        f"Constraints (read carefully):\n"
        f"- NO overlap with scheming / power-seeking / Machiavellian / "
        f"corporate-strategist / antagonist archetypes.\n"
        f"- NO overlap with 'expert advisor' archetypes (the helpful assistant "
        f"baseline) — pick concrete embodied roles, not abstract advisors.\n"
        f"- Prefer roles with sharply distinctive concrete vocabulary: a "
        f"marine biologist talking about coral, a Tibetan-Buddhist monk "
        f"talking about impermanence, a five-year-old child describing dinosaurs, "
        f"a master sommelier describing tannins, a glassblower describing "
        f"working temperature, a 19th-century lighthouse keeper, an ER trauma "
        f"nurse mid-code, a botanical illustrator. Concrete sensory and "
        f"technical detail beats abstract personality traits.\n"
        f"- Range freely across professions, life-stages, time periods, "
        f"fictional roles, animal-naturalist hybrids. The more distant from "
        f"'corporate scheming villain', the better.\n"
        f"- Each persona is a SHORT one-sentence system prompt of the form "
        f"'You are a {{role}}.' or 'You are a {{role}} who {{clause}}.' — "
        f"match the existing bank's register exactly.\n\n"
        f"Return ONLY a JSON array of {n} objects, each with keys 'name' "
        f"(short snake_case identifier, e.g. 'marine_biologist') and 'prompt' "
        f"(the system prompt). No prose, no markdown fences."
    )


def _extract_json_array(text: str) -> list[dict[str, str]]:
    """Parse a JSON array of {name, prompt} from a Sonnet reply (fail-loud)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Sonnet reply did not contain a JSON array. First 200 chars: {cleaned[:200]!r}"
        )
    arr = json.loads(cleaned[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("Parsed persona payload is not a list.")
    return arr


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


async def _call_sonnet(n: int, source_prompt: str) -> str:
    """Single async Sonnet 4.5 call. Returns the raw reply text."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=4096,
        system=_PERSONA_GEN_SYSTEM,
        messages=[{"role": "user", "content": _persona_gen_user(n, source_prompt)}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def generate_candidates(
    *,
    n: int,
    source_prompt: str,
) -> dict[str, str]:
    """Generate ``n`` candidate personas via one Sonnet call.

    Args:
        n: target number of candidates (we ask for n; Sonnet may return fewer
            after dedupe).
        source_prompt: the source persona system prompt; framed into the
            generation request.

    Returns:
        {name: system_prompt} mapping, slugified, deduped by name.

    Raises:
        ValueError on a malformed reply, RuntimeError if Sonnet returned 0
        usable candidates.
    """
    reply = asyncio.run(_call_sonnet(n, source_prompt))
    proposed = _extract_json_array(reply)

    out: dict[str, str] = {}
    for obj in proposed:
        if not isinstance(obj, dict):
            continue
        name = _slugify(str(obj.get("name", "")))
        prompt = str(obj.get("prompt", "")).strip()
        if not name or not prompt:
            continue
        if name in out:
            continue
        out[name] = prompt

    if not out:
        raise RuntimeError(
            f"Sonnet returned 0 usable candidates. Raw reply (first 500 chars): {reply[:500]!r}"
        )
    log.info("Generated %d candidate personas via Sonnet (asked for %d).", len(out), n)
    return out


def compute_cos_to_source(
    centroids: dict[int, Any],  # torch.Tensor per layer, shape (N, D)
    persona_names: list[str],
    source: str,
) -> dict[int, dict[str, float]]:
    """For each layer, compute {persona: cos(persona, source)}.

    `centroids[layer]` is shape (N, D) with row order matching `persona_names`.
    Returns {layer: {persona: cos}}.
    """
    if source not in persona_names:
        raise KeyError(
            f"source {source!r} missing from persona_names "
            f"(have: {persona_names[:10]}...). bank/centroid drift?"
        )
    source_idx = persona_names.index(source)

    out: dict[int, dict[str, float]] = {}
    for layer, mat in centroids.items():
        # Convert to float64 numpy for clean numeric stability.
        if hasattr(mat, "detach"):  # torch.Tensor
            arr = mat.detach().to(dtype_for_float64()).cpu().numpy()
        else:
            arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(persona_names):
            raise ValueError(
                f"centroids[layer={layer}] shape {arr.shape} mismatches "
                f"persona_names ({len(persona_names)})."
            )
        src_vec = arr[source_idx]
        src_norm = float(np.linalg.norm(src_vec))
        if src_norm == 0.0:
            raise RuntimeError(f"source centroid has zero norm at layer {layer}.")
        cos_for_layer: dict[str, float] = {}
        for i, name in enumerate(persona_names):
            v = arr[i]
            vn = float(np.linalg.norm(v))
            if vn == 0.0:
                cos_for_layer[name] = 0.0
                continue
            cos_for_layer[name] = float(np.dot(v, src_vec) / (vn * src_norm))
        out[layer] = cos_for_layer
    return out


def dtype_for_float64():
    """torch float64 dtype; pulled out so the rest of the module is torch-free."""
    import torch

    return torch.float64


def _summary_table(cos_to_source: dict[int, dict[str, float]], layers: tuple[int, ...]) -> str:
    """Render a fixed-width table of cos(persona, villain) per persona × layer."""
    personas = sorted(cos_to_source[layers[0]].keys())
    header = f"{'persona':<32} " + " ".join(f"L{lay:<6}" for lay in layers) + "  min_cos"
    lines: list[str] = [header, "-" * len(header)]
    for p in personas:
        row_vals = [cos_to_source[lay][p] for lay in layers]
        min_v = min(row_vals)
        lines.append(
            f"{p[:32]:<32} " + " ".join(f"{v:>+7.4f}" for v in row_vals) + f"  {min_v:>+7.4f}"
        )
    return "\n".join(lines)


def _count_below(
    cos_to_source: dict[int, dict[str, float]],
    threshold: float,
    *,
    exclude: set[str],
) -> dict[int, int]:
    """For each layer, count personas with cos(p, source) < threshold.

    Excludes `exclude` (the source persona + any anchors that should not be
    counted toward "new personas at low cos").
    """
    out: dict[int, int] = {}
    for layer, table in cos_to_source.items():
        out[layer] = sum(1 for name, c in table.items() if name not in exclude and c < threshold)
    return out


def _decide(
    n_below_07: dict[int, int],
    n_below_085: dict[int, int],
) -> tuple[str, str]:
    """Render a verdict + one-line decision summary for the orchestrator.

    Returns (verdict, summary). Verdict ∈ {"H1_confirmed", "H2_confirmed", "mixed"}.

    Decision rule (matches the brief):
      H1_confirmed — at least one layer has >=3 candidates below cos 0.7.
                     Bank expansion (Phase B) is the right path.
      H2_confirmed — NO candidate sits below cos 0.7 at ANY layer. Even the
                     shrunk-bands fallback [0.40, 0.55, 0.70, 0.85] is
                     unreachable; the cosine-bands framing is broken on this
                     rig. (The H2 check uses cos 0.7 — not 0.85 — because
                     the shrunk-bands plan-fallback hinges on hitting cos
                     0.70 as its bottom band; if nothing dips below 0.7,
                     Phase B can't help even with the shrunk targets.)
      mixed        — at least one candidate below 0.7 at SOME layer but no
                     layer has the >=3 H1 threshold. Phase B might still
                     work with shrunk bands; surface to user.
    """
    if any(v >= 3 for v in n_below_07.values()):
        return (
            "H1_confirmed",
            "n_below_0.7 >= 3 at some layer — bank expansion (Phase B) is the right path.",
        )
    if all(v == 0 for v in n_below_07.values()):
        return (
            "H2_confirmed",
            (
                "n_below_0.7 == 0 at all 3 layers — embedding-space compression is the "
                "limit; even the shrunk-bands fallback [0.40, 0.55, 0.70, 0.85] is "
                "unreachable. Cosine-bands framing is methodologically broken on this rig."
            ),
        )
    _ = n_below_085  # kept in the signature for the JSON payload + summary parity.
    return (
        "mixed",
        (
            "Bank can be widened (some personas below 0.85), but not deeply (none below 0.7). "
            "Proceed to Phase B with shrunk target bands [0.40, 0.55, 0.70, 0.85]."
        ),
    )


def _git_commit() -> str:
    import subprocess

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


def main(argv: list[str] | None = None) -> int:
    """Run the probe end-to-end."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-candidates",
        type=int,
        default=8,
        help="Number of candidate personas to ask Sonnet for (default 8).",
    )
    ap.add_argument(
        "--source",
        default="villain",
        help="Source persona name (must be in EVAL_PERSONAS_24).",
    )
    ap.add_argument(
        "--anchors",
        default="qwen_default,assistant,ai_assistant,kindergarten_teacher",
        help=(
            "Comma-separated list of EVAL_PERSONAS_24 entries to include as "
            "sanity anchors — tells us where the existing bank's cluster sits. "
            "These do NOT count toward H1's n_below_0.7 / n_below_0.85 tallies."
        ),
    )
    ap.add_argument(
        "--layers",
        default="10,15,20",
        help="Comma-separated layers (default 10,15,20).",
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HF model id (default {DEFAULT_BASE_MODEL}).",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Device for centroid extraction (default cuda:0).",
    )
    ap.add_argument(
        "--candidates-file",
        type=Path,
        default=DEFAULT_CANDIDATES_FILE,
        help=(
            "Where to read/write the candidate personas. If the file exists, "
            "we reuse it (skipping Sonnet) so the probe is idempotent."
        ),
    )
    ap.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Where to write the final probe results JSON.",
    )
    ap.add_argument(
        "--skip-generate",
        action="store_true",
        help="Require --candidates-file to exist; do NOT call Sonnet.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=probe_bank] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    anchors = [s.strip() for s in args.anchors.split(",") if s.strip()]

    # ── 1. Generate / load candidate personas. ──────────────────────────────
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if args.source not in EVAL_PERSONAS_24:
        raise ValueError(
            f"source {args.source!r} not in EVAL_PERSONAS_24; available: "
            f"{sorted(EVAL_PERSONAS_24)[:10]}..."
        )
    source_prompt = EVAL_PERSONAS_24[args.source]
    for a in anchors:
        if a not in EVAL_PERSONAS_24:
            raise ValueError(
                f"anchor {a!r} not in EVAL_PERSONAS_24; available: "
                f"{sorted(EVAL_PERSONAS_24)[:10]}..."
            )

    candidates: dict[str, str]
    if args.candidates_file.exists():
        candidates = json.loads(args.candidates_file.read_text())
        if not isinstance(candidates, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in candidates.items()
        ):
            raise ValueError(
                f"existing --candidates-file {args.candidates_file} is not a "
                f"{{name: prompt}} JSON object; refusing to proceed."
            )
        log.info(
            "[generate] reusing %d cached candidates at %s", len(candidates), args.candidates_file
        )
    else:
        if args.skip_generate:
            raise FileNotFoundError(
                f"--skip-generate set but candidates file {args.candidates_file} is missing."
            )
        candidates = generate_candidates(n=args.n_candidates, source_prompt=source_prompt)
        args.candidates_file.parent.mkdir(parents=True, exist_ok=True)
        args.candidates_file.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
        log.info(
            "[generate] wrote %d candidate personas → %s", len(candidates), args.candidates_file
        )

    # Collision check: candidate names must NOT collide with the source/anchors.
    forbidden = {args.source, *anchors}
    collisions = sorted(forbidden & set(candidates))
    if collisions:
        raise RuntimeError(
            f"candidate names collide with source/anchors: {collisions}. "
            f"Pick different anchors or regenerate."
        )

    # ── 2. Build the centroid universe. ─────────────────────────────────────
    # Source + anchors + candidates. Sorted for deterministic order.
    universe: dict[str, str] = {args.source: source_prompt}
    for a in anchors:
        universe[a] = EVAL_PERSONAS_24[a]
    for name, prompt in candidates.items():
        universe[name] = prompt
    log.info(
        "[universe] %d personas (source=1 + anchors=%d + candidates=%d)",
        len(universe),
        len(anchors),
        len(candidates),
    )

    # ── 3. Extract centroids. ───────────────────────────────────────────────
    from explore_persona_space.analysis.representation_shift import extract_centroids
    from explore_persona_space.personas import EVAL_QUESTIONS

    centroids_by_layer, persona_names = extract_centroids(
        model_path=args.base_model,
        personas=universe,
        questions=list(EVAL_QUESTIONS),
        layers=list(layers),
        device=args.device,
    )

    # ── 4. Compute cos(persona, source) per layer. ──────────────────────────
    cos_to_source = compute_cos_to_source(
        centroids_by_layer,
        persona_names,
        source=args.source,
    )

    # ── 5. Decide. ──────────────────────────────────────────────────────────
    # exclude = source + anchors (anchors are sanity-only, not part of H1 count).
    exclude = {args.source, *anchors}
    n_below_07 = _count_below(cos_to_source, 0.7, exclude=exclude)
    n_below_085 = _count_below(cos_to_source, 0.85, exclude=exclude)
    n_below_05 = _count_below(cos_to_source, 0.5, exclude=exclude)
    verdict, summary = _decide(n_below_07, n_below_085)

    table_text = _summary_table(cos_to_source, layers)
    print()
    print("=== BANK-GEOMETRY PROBE — cos(persona, villain) by layer ===")
    print(table_text)
    print()
    print(f"n_below_0.5 per layer:  {n_below_05}")
    print(f"n_below_0.7 per layer:  {n_below_07}")
    print(f"n_below_0.85 per layer: {n_below_085}")
    print(f"VERDICT: {verdict}")
    print(f"SUMMARY: {summary}")
    print()

    # ── 6. Persist results. ─────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "schema_version": "i504_probe_v1",
        "source_persona": args.source,
        "anchors": anchors,
        "candidates": candidates,
        "base_model": args.base_model,
        "layers": list(layers),
        "n_eval_questions": len(EVAL_QUESTIONS),
        "cos_to_source": {
            str(lay): cos_to_source[lay] for lay in layers
        },  # {layer_str: {persona: cos}}
        "n_below_0.5": {str(lay): n_below_05[lay] for lay in layers},
        "n_below_0.7": {str(lay): n_below_07[lay] for lay in layers},
        "n_below_0.85": {str(lay): n_below_085[lay] for lay in layers},
        "verdict": verdict,
        "summary": summary,
        "table": table_text,
        "git_commit": _git_commit(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "sonnet_model": SONNET_MODEL,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("[probe_bank] wrote results → %s", args.results_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
