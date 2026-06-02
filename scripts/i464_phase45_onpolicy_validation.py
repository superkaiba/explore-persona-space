"""Phase 4.5 — on-policy validation (MF-B(2) check).

Issue #464 plan v2 §4.1 Phase 4.5. For each of the 9 trained LoRAs,
generate the model's OWN greedy response under that arm's own encoding
for a 16-question subsample of Q_test (per persona), then compute the
mean character-level edit-distance vs the frozen R_canon (which was
generated under the SYSTEM encoding by the BASE model).

Decision rule (plan §4.1):
  edit_distance(role) / edit_distance(system_plain) > 1.5
    → switch Phase 5 headline to trained-greedy on-policy R
      (re-run Phase 4 with arm-specific R).
  else → report ratio as a diagnostic; keep R_canon headline.

Writes ``eval_results/issue_464/onpolicy_validation.json`` with per-arm
mean / median character edit-distances + the switch flag.

CLI:
    uv run python scripts/i464_phase45_onpolicy_validation.py
    uv run python scripts/i464_phase45_onpolicy_validation.py --n-q 4 \
        --smoke-cells system_plain_seed42
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("i464.phase45")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"
LOCAL_DATA_DIR = Path("data/issue_464")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i464")
OUT_PATH = Path("eval_results/issue_464/onpolicy_validation.json")
# Round-2 fix (review blocker #8): per-cell JSONs persisted incrementally
# so a crash mid-sweep doesn't lose prior validation work (CLAUDE.md
# "checkpoint per phase" rule).
PER_CELL_DIR = Path("eval_results/issue_464/onpolicy_validation/per_cell")
SWITCH_THRESHOLD = 1.5

SEEDS = (42, 137, 1337)


def _load_R_canon_test() -> dict[str, dict[str, dict]]:
    """Load R_canon_test from disk; HF fallback."""
    local = LOCAL_DATA_DIR / "R_canon_test.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    return json.loads(local.read_text())["completions"]


def _download_adapter(arm: enc.Arm, seed: int) -> str:
    """Download the (arm, seed) adapter from HF if not cached locally."""
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i464_{arm}_seed{seed}"
    local_target = LOCAL_ADAPTER_CACHE / target_subpath
    local_target.mkdir(parents=True, exist_ok=True)
    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            revision="main",
            filename=f"{target_subpath}/{fname}",
            local_dir=LOCAL_ADAPTER_CACHE,
        )
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local_target}")
    return str(local_target)


def _char_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (character-level)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _compute_onpolicy_switch_inputs(
    per_arm_summary: dict[str, dict],
) -> tuple[float | None, float | None]:
    """Return (role_mean, system_plain_mean) edit-distances for the switch rule.

    Extracted as a pure helper (round-4 review blocker #1) so the
    zero-denominator branch in ``_onpolicy_switch_verdict`` is testable
    without invoking the whole CLI.
    """
    role_mean = per_arm_summary.get("role", {}).get("mean")
    plain_mean = per_arm_summary.get("system_plain", {}).get("mean")
    return role_mean, plain_mean


def _onpolicy_switch_verdict(
    role_mean: float | None,
    plain_mean: float | None,
    switch_threshold: float,
) -> tuple[float | None, bool]:
    """Decide (ratio, switch) for the MF-B(2) on-policy validation.

    Round-4 fix (review blocker #1). The round-2/3 logic only set
    ``switch=True`` when ``plain_mean > 0 AND role_mean/plain_mean >
    threshold``. But R_canon is generated under the SYSTEM encoding, so
    the system_plain arm's trained-greedy R is ~identical to R_canon ->
    plain_mean ~ 0 in the normal case. The zero-denominator branch then
    silently disabled the safeguard in exactly the high-role-drift case
    it exists to catch. Round-4: treat ``plain_mean == 0 AND role_mean
    > 0`` as ``ratio = inf, switch = True``; keep ``switch = False``
    only when BOTH means are 0 (no drift detected anywhere) or either
    is None (missing data, can't decide).

    Returns ``(ratio, switch)`` where:
      - ``ratio`` is ``role_mean / plain_mean`` if plain_mean > 0,
        ``float('inf')`` if plain_mean == 0 and role_mean > 0,
        ``None`` otherwise (both 0, or either is None).
      - ``switch`` is True iff role-arm divergence exceeds the
        threshold relative to system_plain.
    """
    if role_mean is None or plain_mean is None:
        return None, False
    if plain_mean > 0:
        ratio = role_mean / plain_mean
        return ratio, ratio > switch_threshold
    # plain_mean == 0 (system arm matches R_canon byte-for-byte).
    if role_mean > 0:
        # Role arm drifts but system_plain doesn't -- ratio is +inf,
        # switch is unambiguously True (this is exactly the MF-B(2)
        # case the rule was designed to catch).
        return float("inf"), True
    # Both means are 0: no drift in either arm, no switch needed.
    return 0.0, False


def main(argv: list[str] | None = None) -> None:
    """Entry point for Phase 4.5 on-policy validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n-q",
        type=int,
        default=16,
        help="Per-persona subsample of Q_test (plan default 16).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument(
        "--smoke-cells",
        nargs="+",
        default=None,
        help="Restrict cells (e.g. 'system_plain_seed42'); smoke use.",
    )
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()

    cells: list[tuple[enc.Arm, int]] = [(arm, seed) for arm in enc.ARMS for seed in SEEDS]
    if args.smoke_cells:
        wanted = set(args.smoke_cells)
        cells = [(a, s) for (a, s) in cells if f"{a}_seed{s}" in wanted]
        logger.warning("SMOKE: restricted to %d cells", len(cells))

    qs = q_test[: args.n_q]
    logger.info(
        "Phase 4.5: %d cells x %d q x 2 personas = %d generations",
        len(cells),
        len(qs),
        len(cells) * len(qs) * 2,
    )

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Round-2 fix (review blocker #8): persist per-cell JSON incrementally
    # so a crash on cell N doesn't lose work for cells 0..N-1.
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    per_cell: dict[str, dict] = {}
    per_arm_distances: dict[str, list[float]] = {a: [] for a in enc.ARMS}

    for arm, seed in cells:
        cell_label = f"{arm}_seed{seed}"
        cell_path = PER_CELL_DIR / f"{cell_label}.json"
        # If the per-cell JSON already exists (resume after crash), load it
        # instead of re-running the generation.
        if cell_path.exists() and cell_path.stat().st_size > 0:
            cached = json.loads(cell_path.read_text())
            per_cell[cell_label] = cached
            # Repopulate per_arm_distances from cached per-q ratios so
            # the aggregate matches a fresh run.
            per_arm_distances[arm].extend(cached.get("per_q_ratios", []))
            logger.info("cell=%s loaded from %s (resume)", cell_label, cell_path)
            continue

        adapter_path = _download_adapter(arm, seed)
        lora_req = LoRARequest(
            lora_name=cell_label,
            lora_int_id=cells.index((arm, seed)) + 1,
            lora_path=adapter_path,
        )

        # Build own-encoding prompts for each persona.
        prompts: list[str] = []
        labels: list[tuple[str, str]] = []  # (persona, q)
        for persona in enc.PERSONAS:
            e_eval: enc.EvalEncoding = (
                f"role_{persona}" if arm == "role" else f"system_{persona}"  # type: ignore[assignment]
            )
            for q in qs:
                prompts.append(enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer))
                labels.append((persona, q))

        outs = llm.generate(prompts, sp, lora_request=lora_req)
        cell_dists: list[float] = []
        per_persona_dists: dict[str, list[float]] = {p: [] for p in enc.PERSONAS}
        for (persona, q), out in zip(labels, outs, strict=True):
            r_trained = out.outputs[0].text
            r_canon = R_canon_test[persona][q]["response_text"]
            # Char-level edit-distance, normalized by max length to make
            # cell sizes comparable (plan §6.1 reports mean char ed).
            ed = _char_edit_distance(r_trained, r_canon)
            denom = max(len(r_trained), len(r_canon), 1)
            ratio = ed / denom
            cell_dists.append(ratio)
            per_persona_dists[persona].append(ratio)

        cell_payload = {
            "arm": arm,
            "seed": seed,
            "mean_norm_edit_distance": statistics.mean(cell_dists),
            "median_norm_edit_distance": statistics.median(cell_dists),
            "per_persona_mean": {
                p: statistics.mean(per_persona_dists[p]) if per_persona_dists[p] else None
                for p in enc.PERSONAS
            },
            "n_q": len(qs),
            "per_q_ratios": cell_dists,
        }
        per_cell[cell_label] = cell_payload
        per_arm_distances[arm].extend(cell_dists)
        # Atomic per-cell write (.tmp -> rename) so a crash mid-flush
        # leaves the prior version intact.
        tmp = cell_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cell_payload))
        tmp.replace(cell_path)
        logger.info(
            "cell=%s mean_norm_ed=%.3f median=%.3f -> %s",
            cell_label,
            cell_payload["mean_norm_edit_distance"],
            cell_payload["median_norm_edit_distance"],
            cell_path,
        )

    per_arm_summary = {}
    for arm in enc.ARMS:
        ds = per_arm_distances[arm]
        per_arm_summary[arm] = {
            "n": len(ds),
            "mean": statistics.mean(ds) if ds else None,
            "median": statistics.median(ds) if ds else None,
        }

    # Switch rule: edit_distance(role) / edit_distance(system_plain) > 1.5?
    #
    # Round-4 fix (review blocker #1): the round-2/3 condition
    # `plain_mean > 0` silently DISABLES the MF-B(2) safeguard in its
    # most important case. R_canon is generated under the SYSTEM
    # encoding, so system_plain's trained-greedy R is ~identical to
    # R_canon (mean edit-distance ~0 in the normal training regime).
    # So when the role arm DOES drift (role_mean > 0) the original
    # guard reads `plain_mean == 0` and silently sets `switch=False`
    # — exactly the high-role-drift case the gate exists to catch.
    # Treat zero plain + non-zero role as infinite ratio + switch=True.
    role_mean, plain_mean = _compute_onpolicy_switch_inputs(per_arm_summary)
    ratio, switch = _onpolicy_switch_verdict(role_mean, plain_mean, SWITCH_THRESHOLD)

    # JSON doesn't natively encode float('inf'); serialize as the string
    # "inf" so the consumer (Phase 5) sees a recognizable token. The
    # downstream uses (Phase 5's switch decision + plot label) are
    # type-tolerant: switch_headline_to_trained_R is the load-bearing bool.
    import math as _math

    ratio_for_json: float | str | None = ratio
    if isinstance(ratio, float) and _math.isinf(ratio):
        ratio_for_json = "inf"

    payload = {
        "schema_version": "i464_onpolicy_validation_v1",
        "switch_threshold": SWITCH_THRESHOLD,
        "n_q_per_persona": args.n_q,
        "per_cell": per_cell,
        "per_arm": per_arm_summary,
        "role_over_system_plain_ratio": ratio_for_json,
        "switch_headline_to_trained_R": switch,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Phase 4.5 done. role/system_plain edit_ratio=%s switch=%s -> %s",
        ratio,
        switch,
        OUT_PATH,
    )


if __name__ == "__main__":
    main()
