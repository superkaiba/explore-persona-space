#!/usr/bin/env python3
"""Materialize ``base_cosines.json`` for the #519/#521 Phase D regression.

Round-2 reviewer ``missing-base-cosines-hook`` fix. The dispatcher
(``scripts/issue_519_dispatch.py``) accepts ``--base-cosines-json
PATH``; when passed it computes the Mechanism-A test (Spearman of
``||Delta_v_b(c)||`` against ``cos_base(source, c)``) inside Phase D's
SVD analyzer. v1 launch flow expected the file to exist but provided
no script to build it; the dispatcher's silent-skip on missing file
let the headline metric ship as null.

What this script writes
-----------------------

A JSON ``{persona_name: float}`` covering every persona in the panel,
where the float is the cosine similarity (canonical persona-vectors
recipe (a) — last prompt token, per
``.claude/rules/persona-distance-metrics.md``) in the BASE
Qwen-2.5-7B-Instruct residual stream at layer L=14 between:

- the source persona's last-prompt-token activation, and
- the panel persona's last-prompt-token activation,

averaged over the eval-question pool. The source persona is by default
``medical_doctor`` (#519/#521 source); override with ``--source``.

By construction the source persona's own entry is 1.0.

CLI
---

::

    # Production: full panel, GPU-bound, ~1 min on a single H100.
    uv run python scripts/issue_521_build_base_cosines.py \\
        --personas-json eval_results/issue_521/inputs/personas.json \\
        --questions-json eval_results/issue_521/inputs/questions.json \\
        --layer 14 \\
        --source medical_doctor \\
        --output-dir eval_results/issue_521/inputs

    # Smoke (CPU-only, mocked activations — verifies plumbing).
    uv run python scripts/issue_521_build_base_cosines.py --tiny \\
        --personas-json /tmp/issue-521-inputs/personas.json \\
        --output-dir /tmp/issue-521-base-cosines

GPU-bound carve-out
-------------------

The production path requires loading the BASE Qwen-2.5-7B-Instruct
model (~14 GB bf16), so it cannot run on a CPU-only VM. The ``--tiny``
mode short-circuits the HF load and generates deterministic random
activation vectors per persona (seeded by persona name) so the
plumbing — output dir, JSON shape, presence of the source entry as
1.0, persona-set coverage — can be exercised end-to-end on a local
machine without a GPU. The production launcher (Phase 4 of the #521
launch sequence per the round-2 reviewer hook) runs the GPU path on
the pod between ``stage_adapters`` and the dispatcher invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYER = 14
DEFAULT_SOURCE = "medical_doctor"


def _resolve_repo_root() -> Path:
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mock_persona_activation(persona_name: str, hidden_size: int = 64) -> list[float]:
    """Deterministic random activation for --tiny mode (no HF load).

    Seeded by SHA-1(persona_name) so the same persona always maps to
    the same vector — the smoke can re-check cosine reproducibility.
    """
    import numpy as np

    h = hashlib.sha1(persona_name.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(hidden_size).astype("float32").tolist()


def _compute_real_activations(
    *,
    base_model_name: str,
    personas: dict[str, str],
    questions: Sequence[str],
    layer: int,
) -> dict[str, list[float]]:
    """Load BASE Qwen-2.5-7B-Instruct, forward each persona's prompts,
    return per-persona mean-over-questions last-prompt-token activation.

    Recipe (a) per ``.claude/rules/persona-distance-metrics.md``: take
    the residual-stream activation at the final input position (the
    last prompt token) of ``{persona, question}`` ChatML, average
    across the question pool.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    logger.info("[phase=load_base] loading %s ...", base_model_name)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=False,  # set per-call instead
    )
    model.eval()
    logger.info("[phase=load_base_done] loaded in %.1fs", time.time() - t0)

    activations: dict[str, list[float]] = {}
    for persona_name, persona_prompt in personas.items():
        per_q_vecs = []
        for q in questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # hidden_states[layer + 1] = output of block `layer` (post-residual).
            h = out.hidden_states[layer + 1]
            assert h.dim() == 3, f"expected (B, T, H), got {h.shape}"
            # Last prompt token = slot -1 of the encoded text (no response yet).
            per_q_vecs.append(h[0, -1].detach().float().cpu().numpy())
        import numpy as np

        mean_vec = np.mean(np.stack(per_q_vecs, axis=0), axis=0)
        activations[persona_name] = mean_vec.tolist()
        logger.info(
            "[persona=%s] activation built over %d questions (||a||=%.3f)",
            persona_name,
            len(questions),
            float(np.linalg.norm(mean_vec)),
        )
    return activations


def build_base_cosines(
    *,
    personas: dict[str, str],
    questions: Sequence[str],
    source: str,
    layer: int,
    base_model_name: str,
    tiny: bool,
) -> dict[str, float]:
    """Top-level: return ``{persona_name: cos_base(source, persona)}``.

    Source persona's own entry is 1.0 by construction.
    """
    if source not in personas:
        raise KeyError(
            f"source persona {source!r} not present in personas.json; "
            f"known keys: {sorted(personas.keys())!r}"
        )

    if tiny:
        logger.info("[phase=mock_activations] --tiny: using deterministic mock vectors")
        activations: dict[str, list[float]] = {
            name: _mock_persona_activation(name) for name in personas
        }
    else:
        activations = _compute_real_activations(
            base_model_name=base_model_name,
            personas=personas,
            questions=questions,
            layer=layer,
        )

    src_act = activations[source]
    cosines: dict[str, float] = {}
    for name in personas:
        cosines[name] = _cosine(src_act, activations[name])
    # By construction the source's own cosine is 1.0 (within floating-
    # point — assert).
    assert abs(cosines[source] - 1.0) < 1e-5, (
        f"source self-cosine should be ~1.0, got {cosines[source]}"
    )
    return cosines


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build base_cosines.json for #521 Phase D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--personas-json",
        required=True,
        help="Path to JSON {persona_name: system_prompt} (issue_521_build_inputs output).",
    )
    p.add_argument(
        "--questions-json",
        default=None,
        help=(
            "Path to JSON list[str] of eval questions. Required unless "
            "--tiny is set (mock mode doesn't forward through a model)."
        ),
    )
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source persona name (default: {DEFAULT_SOURCE!r}).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Residual-stream layer for the last-prompt-token read (default: {DEFAULT_LAYER}).",
    )
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HF model id for the BASE (untrained) model (default: {DEFAULT_BASE_MODEL!r}).",
    )
    p.add_argument(
        "--output-dir",
        default="eval_results/issue_521/inputs",
        help="Directory to write base_cosines.json into.",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Smoke-mode: skip the HF model load and use deterministic "
            "mock activations seeded by persona name. Lets the plumbing "
            "(JSON shape, source-self-cosine = 1.0, full persona "
            "coverage) be exercised on a CPU-only VM."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = _resolve_repo_root()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    personas_path = Path(args.personas_json)
    if not personas_path.is_absolute():
        personas_path = repo_root / personas_path
    if not personas_path.exists():
        raise FileNotFoundError(
            f"--personas-json={args.personas_json!r} not found at {personas_path}"
        )
    personas = json.loads(personas_path.read_text())
    logger.info("loaded %d personas from %s", len(personas), personas_path)

    if args.tiny:
        # Questions are unused in tiny mode (no forward pass).
        questions: list[str] = []
    else:
        if args.questions_json is None:
            raise ValueError(
                "--questions-json is required for the production path "
                "(omit only with --tiny smoke mode)."
            )
        q_path = Path(args.questions_json)
        if not q_path.is_absolute():
            q_path = repo_root / q_path
        questions = json.loads(q_path.read_text())
        logger.info("loaded %d questions from %s", len(questions), q_path)

    cosines = build_base_cosines(
        personas=personas,
        questions=questions,
        source=args.source,
        layer=args.layer,
        base_model_name=args.base_model,
        tiny=args.tiny,
    )

    # Reproducibility metadata: written to a companion `_meta.json` so
    # `base_cosines.json` stays a flat {name: float} that Phase D can
    # `json.load` without unwrapping.
    out_path = out_dir / "base_cosines.json"
    out_path.write_text(json.dumps(cosines, indent=2, sort_keys=True))
    logger.info("wrote %s (N=%d personas, source=%s)", out_path, len(cosines), args.source)

    try:
        import subprocess

        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    meta = {
        "source": args.source,
        "layer": args.layer,
        "base_model": args.base_model,
        "n_personas": len(cosines),
        "n_questions": len(questions),
        "tiny": args.tiny,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "personas_json": str(personas_path),
        "questions_json": (str(q_path) if not args.tiny else None),
    }
    meta_path = out_dir / "base_cosines_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("[phase=done] wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
