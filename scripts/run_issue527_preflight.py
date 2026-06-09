"""Issue #527 Phase 0 preflight — CPU-runnable, pre-pod-provision (plan §4 Step 0).

HARD GATE for the orchestrator (`/issue` Step 6d.0). Fails LOUD with a clear
message on the FIRST missing piece. No pod is provisioned until this passes.

Checks (plan §4 Step 0 + §11 / §12):
  1. Persona-registry resolution: every persona referenced (19-persona pool,
     4 contrastive-negative panel, fallback chain) resolves in
     ``data/issue_472/persona_bank.json``.
  2. Marker token assert: ``tokenizer.encode(" ※", add_special_tokens=False)
     == [83399]`` on the canonical Qwen-2.5-7B-Instruct tokenizer.
  3. ``<|im_end|>`` id assert (must equal 151645).
  4. AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct") sanity load
     (config-only, no weights).
  5. Question-pool reachability (smoke-fallback ALLOWED only with
     ``--allow-smoke-fallback`` — the main pipeline path forbids it).
  6. HF Hub auth sanity (HF_TOKEN env var set + ``list_repo_files`` on the
     model repo succeeds — confirms upload-policy preconditions).

CLI:
    uv run python scripts/run_issue527_preflight.py
    uv run python scripts/run_issue527_preflight.py --allow-smoke-fallback
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import logging
import os
import sys

from explore_persona_space.experiments.issue_527 import (
    BASE_MODEL,
    HF_MODEL_REPO,
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.issue_527.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.experiments.issue_527.question_pool import load_question_pool

log = logging.getLogger("issue_527.preflight")


def main(argv: list[str] | None = None) -> int:
    # `uv run python` does NOT auto-load `.env`; load it BEFORE any HF_TOKEN-dependent call.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--allow-smoke-fallback",
        action="store_true",
        help=(
            "Permit a smoke-sized question-pool fallback (20 EVAL_QUESTIONS). "
            "ONLY for smoke runs; main pipeline must omit this flag."
        ),
    )
    ap.add_argument(
        "--n-questions",
        type=int,
        default=400,
        help="Number of questions to require from the question pool (default 400).",
    )
    ap.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip the tokenizer-based marker/im_end asserts (only useful in unit tests).",
    )
    args = ap.parse_args(argv)

    # Step 1: persona-registry hard gate.
    log.info("Step 1: persona-registry resolution against data/issue_472/persona_bank.json")
    personas = load_persona_bank()
    assert_registry_resolves(personas)
    log.info("Step 1 PASS: %d personas resolved.", len(personas))

    # Steps 2 + 3: marker token + im_end ids.
    if not args.skip_tokenizer:
        log.info("Step 2: marker token tokenizes to MARKER_ID=%d", MARKER_ID)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [MARKER_ID]:
            raise AssertionError(
                f"MARKER token drift: encode({MARKER_TEXT!r}) -> {ids}, "
                f"expected [{MARKER_ID}]. Refusing to advance — this would "
                "silently train the wrong token (bare ※ id 63680 vs leading-space "
                f" ※ id {MARKER_ID})."
            )
        log.info("Step 2 PASS.")

        log.info("Step 3: <|im_end|> token id assert (IM_END_ID=%d)", IM_END_ID)
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end != IM_END_ID:
            raise AssertionError(
                f"<|im_end|> id drift: got {im_end}, expected {IM_END_ID}. "
                "Refusing to advance — the post-response slot resolver would "
                "look for the wrong terminator."
            )
        log.info("Step 3 PASS.")
    else:
        log.warning("Skipping tokenizer asserts (--skip-tokenizer)")

    # Step 4: model-config sanity (no weights download).
    log.info("Step 4: AutoConfig sanity for %s", BASE_MODEL)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if cfg.hidden_size != 3584:
        raise AssertionError(f"hidden_size drift: cfg.hidden_size={cfg.hidden_size}, expected 3584")
    if cfg.num_hidden_layers < 21:
        raise AssertionError(
            f"num_hidden_layers={cfg.num_hidden_layers} — L20 extraction needs "
            "at least 21 layers (indices 0..20)."
        )
    log.info(
        "Step 4 PASS: hidden_size=%d num_hidden_layers=%d",
        cfg.hidden_size,
        cfg.num_hidden_layers,
    )

    # Step 5: question-pool reachability.
    log.info("Step 5: question-pool reachability (n=%d)", args.n_questions)
    qs = load_question_pool(
        n_required=args.n_questions, allow_smoke_fallback=args.allow_smoke_fallback
    )
    log.info("Step 5 PASS: loaded %d questions (first 2: %s)", len(qs), qs[:2])

    # Step 6: HF Hub auth + reachability.
    log.info("Step 6: HF Hub auth + reachability against %s", HF_MODEL_REPO)
    if not os.environ.get("HF_TOKEN"):
        raise AssertionError(
            "HF_TOKEN missing from environment — uploads will fail at training "
            "time (`upload_model` returns empty path → silent loss). "
            "Source `.env` or set HF_TOKEN before retrying."
        )
    try:
        from huggingface_hub import list_repo_files

        _ = list_repo_files(HF_MODEL_REPO, revision="main")
        log.info("Step 6 PASS.")
    except Exception as e:
        raise RuntimeError(
            f"HF Hub reachability check failed against {HF_MODEL_REPO}: {e}. "
            "Adapter upload would fail later; refusing to provision a pod."
        ) from e

    log.info("ALL PREFLIGHT CHECKS PASSED — proceed with Phase A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
