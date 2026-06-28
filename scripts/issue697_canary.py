"""Issue #697 — pre-sweep canary (Gate C1 + Gate C2). MUST PASS before the sweep.

The patch hook (``analysis.cv_patch``) is the linchpin of #697's whole result, so
a cheap pre-sweep canary rules out a wasted sweep on a broken hook (plan §7).

Gate C1 — patch correctness (three orthogonal asserts on the production panel
through ``cv_patch``):
  C1.1  Self-patch identity ≈ 0 in BOTH read mode AND generate mode (a patch with
        the model's OWN context residual is an exact no-op up to fp tolerance).
  C1.2  NON-IDENTITY KV-cache propagation: a non-identity patch (a) moves the
        first-token logits vs unpatched by > eps (the patch propagates through
        KV-cached decoding), AND (b) the use_cache=True / use_cache=False
        first-token logits agree within 1e-3 (caching does not drop the patch).
        DIVERGENCE on (b) => production falls back to use_cache=False (no HALT).
  C1.3  Decoded-token slot audit: ``content_patch_pos`` on a real panel
        (persona, question) lands on a content token; ``audit_patch_slot`` does
        not raise. FAIL => the slot regressed onto a header token => HALT.

Gate C2 — rsLoRA application-scaling parity (the inherited #651 Gate 7a):
  reproduce #521's committed marker numbers (``same_marker_seed42.json``:
  s_top1_frac 0.32465, mean_cos_to_U1 0.58711, cos(U1) >= 0.95 within tol 0.05)
  through the SAME ``merge_and_unload`` path the sweep uses — proves the rsLoRA
  alpha/sqrt(r) read gauge matches the parent committed regime (#601) AND the adapter
  applies. Delegates to the vendored ``issue651_canary.gate_7a``.

Smoke mode (``--cpu-only --smoke-model <tiny>``): Gate C1 runs on the tiny model
(the cv_patch invariants are model-agnostic); Gate C2 is SKIPPED (it needs the
real 7B marker adapter + #521's committed numbers, which a tiny model cannot
reproduce) and reported as ``skipped (smoke)``. The real GPU canary runs BOTH.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from explore_persona_space.analysis import cv_patch

logger = logging.getLogger("issue697_canary")

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
EPS_LOGIT_MOVE = 1e-4
KV_PARITY_TOL = 1e-3
SELF_PATCH_TOL = 1e-3


def _repo_root() -> Path:
    import subprocess

    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


def _panel_probe():
    """A real (system_prompt, question) probe from the fixed #651 panel."""
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    personas = build_panel_personas()
    questions = build_panel_questions()
    # the first non-assistant persona + first question.
    p_name = next(iter(personas))
    return personas[p_name], questions[0], p_name


def gate_c1(*, cpu_only: bool, smoke_model: str | None) -> dict:
    """cv_patch correctness on a real model + the production panel (C1.1/1.2/1.3)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = smoke_model or QWEN_ID
    logger.info("[phase=gate_c1] model=%s cpu_only=%s", model_id, cpu_only)
    device_map = None if cpu_only else "auto"
    dtype = torch.float32 if cpu_only else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    ).eval()

    system_prompt, question, p_name = _panel_probe()
    layer = 1 if smoke_model else 14  # tiny model has few layers; 14 needs the 7B
    n_layers = model.config.num_hidden_layers
    layer = min(layer, n_layers - 1)

    # Build the real ChatML prompt (add_generation_prompt=True — the forward recipe).
    full = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = torch.tensor(tok(full, add_special_tokens=False).input_ids)

    # ---- C1.3 decoded-token slot audit (do this FIRST — cheap, HALTs early) ----
    patch_pos = cv_patch.content_patch_pos(tok, system_prompt, question)
    cv_patch.audit_patch_slot(tok, prompt_ids, patch_pos)  # raises SlotAuditError on a bad slot
    decoded = tok.decode([int(prompt_ids[patch_pos])], skip_special_tokens=False)
    logger.info(
        "[phase=gate_c1.3] PASS: patch_pos=%d decodes to %r (persona=%s)",
        patch_pos,
        decoded,
        p_name,
    )

    # ---- C1.1 self-patch identity (read + generate) ----
    response_start = prompt_ids.shape[0]  # treat the whole prompt as the "response" tail for read
    # capture the model's own layer-L residual at patch_pos.
    with torch.no_grad():
        out = model(prompt_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
    own_cv = out.hidden_states[layer + 1][0, patch_pos].clone()

    unpatched_read = cv_patch.patched_read(
        model, prompt_ids, layer, [], None, max(response_start - 1, 1)
    )
    self_patch_read = cv_patch.patched_read(
        model, prompt_ids, layer, [patch_pos], own_cv, max(response_start - 1, 1)
    )
    read_deltas = {
        k: float((unpatched_read[k] - self_patch_read[k]).abs().max())
        for k in ("mean_resp", "slot")
    }
    c1_1_read_ok = all(d < SELF_PATCH_TOL for d in read_deltas.values())

    unpatched_gen = cv_patch.patched_generate(
        model, tok, prompt_ids, layer, [], None, max_new_tokens=5, do_sample=False
    )
    self_patched_gen = cv_patch.patched_generate(
        model, tok, prompt_ids, layer, [patch_pos], own_cv, max_new_tokens=5, do_sample=False
    )
    c1_1_gen_ok = self_patched_gen == unpatched_gen
    logger.info("[phase=gate_c1.1] self-patch read Δ=%s gen_match=%s", read_deltas, c1_1_gen_ok)

    # ---- C1.2 non-identity KV-cache propagation ----
    # Non-identity donor: a random-norm-matched CV at the slot.
    g = torch.randn_like(own_cv)
    donor = g / torch.linalg.norm(g) * torch.linalg.norm(own_cv)
    logits_unpatched = cv_patch.first_token_logits(
        model, prompt_ids, layer, [], None, use_cache=True
    )
    logits_patched = cv_patch.first_token_logits(
        model, prompt_ids, layer, [patch_pos], donor, use_cache=True
    )
    moved = float((logits_unpatched - logits_patched).abs().max())
    c1_2a_ok = moved > EPS_LOGIT_MOVE
    logits_cache = logits_patched
    logits_nocache = cv_patch.first_token_logits(
        model, prompt_ids, layer, [patch_pos], donor, use_cache=False
    )
    parity = float((logits_cache - logits_nocache).abs().max())
    c1_2b_ok = parity < KV_PARITY_TOL
    # Production use_cache: True only when parity is COMFORTABLY below tol. A
    # "marginal pass" — parity within the top decile of tol [tol/10, tol) — flips
    # to uncached as the safety net (brief concern #4), so a borderline KV path
    # never silently runs the sweep cached. A FAIL (parity >= tol) is also
    # uncached (caching drops the patch).
    marginal = KV_PARITY_TOL / 10.0
    use_cache_default = parity < marginal
    logger.info(
        "[phase=gate_c1.2] patch moves logits by %.3e (>%.1e: %s); cache-vs-nocache Δ=%.3e "
        "(<%.1e: %s; marginal<%.1e); production use_cache=%s",
        moved,
        EPS_LOGIT_MOVE,
        c1_2a_ok,
        parity,
        KV_PARITY_TOL,
        c1_2b_ok,
        marginal,
        use_cache_default,
    )

    checks = {
        "c1_1_self_patch_read": c1_1_read_ok,
        "c1_1_self_patch_generate": c1_1_gen_ok,
        "c1_2a_nonidentity_moves_logits": c1_2a_ok,
        # c1_2b is NOT a HALT condition — divergence flips use_cache, no HALT.
    }
    result = {
        "gate": "C1",
        "model": model_id,
        "layer": layer,
        "patch_pos": patch_pos,
        "patch_pos_decoded": decoded,
        "self_patch_read_deltas": read_deltas,
        "nonidentity_logit_move": moved,
        "kv_cache_parity_delta": parity,
        "use_cache_production_default": use_cache_default,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not result["pass"]:
        raise AssertionError(
            f"GATE C1 FAILED: {checks}. read Δ={read_deltas}, gen_match={c1_1_gen_ok}, "
            f"logit_move={moved:.3e}. HALT before the sweep (plan §7)."
        )
    logger.info("[phase=gate_c1] PASS")
    return result


def gate_c2(*, cpu_only: bool, smoke_model: str | None, max_new_tokens: int) -> dict:
    """rsLoRA parity — the inherited #651 Gate 7a (reproduce #521's committed numbers)."""
    if smoke_model is not None:
        logger.info(
            "[phase=gate_c2] SKIPPED (smoke): needs the real 7B marker adapter + #521's "
            "committed numbers; a tiny model cannot reproduce them."
        )
        return {"gate": "C2", "pass": True, "skipped": "smoke", "model": smoke_model}
    # Import the vendored #651 canary robustly: when this file runs as a SCRIPT
    # (uv run python scripts/issue697_canary.py), sys.path[0] is scripts/ so the
    # bare module name resolves; under a package import the `scripts.` form does.
    try:
        from issue651_canary import gate_7a  # script-mode (sys.path[0]=scripts/)
    except ModuleNotFoundError:
        from scripts.issue651_canary import gate_7a  # package-mode (repo-root cwd)

    repo_root = _repo_root()
    res = gate_7a(repo_root, cpu_only=cpu_only, max_new_tokens=max_new_tokens)
    return {"gate": "C2", "pass": res["pass"], "inherited_gate_7a": res}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--smoke-model",
        default=None,
        help="Tiny base model id for a CPU smoke (e.g. Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--gate", choices=["C1", "C2", "both"], default="both", help="Which gate(s) to run."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()

    out: dict = {}
    if args.gate in ("C1", "both"):
        out["gate_c1"] = gate_c1(cpu_only=args.cpu_only, smoke_model=args.smoke_model)
    if args.gate in ("C2", "both"):
        out["gate_c2"] = gate_c2(
            cpu_only=args.cpu_only,
            smoke_model=args.smoke_model,
            max_new_tokens=args.max_new_tokens,
        )

    repo_root = _repo_root()
    out_dir = repo_root / "eval_results" / "issue_697" / "canary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canary_results.json").write_text(json.dumps(out, indent=2, default=float))
    logger.info("[phase=canary_done] wrote %s", out_dir / "canary_results.json")

    # --- use_cache decision sentinel (concern #4) ---------------------------
    # The dispatcher's sweep phase reads this on startup and threads
    # --use-cache/--no-use-cache to every cell. Gate C1.2's HALT path already
    # aborts a broken hook BEFORE the sweep (c1_2a / self-patch FAIL raise in
    # gate_c1); this sentinel carries the C1.2b parity decision: True when caching
    # does NOT drop the patch, False (run uncached, the safety net) otherwise.
    if "gate_c1" in out:
        decision = bool(out["gate_c1"].get("use_cache_production_default", True))
        write_use_cache_decision(repo_root, decision, out["gate_c1"])
    return 0


def use_cache_decision_path(repo_root: Path) -> Path:
    """Canonical path for the canary→sweep use_cache decision (concern #4)."""
    return repo_root / "eval_results" / "issue_697" / "canary" / "canary_decision.json"


def write_use_cache_decision(repo_root: Path, use_cache: bool, gate_c1: dict) -> Path:
    """Persist the canary's Gate C1.2 use_cache decision for the dispatcher to read."""
    p = use_cache_decision_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "use_cache_production_default": use_cache,
                "kv_cache_parity_delta": gate_c1.get("kv_cache_parity_delta"),
                "kv_parity_tol": KV_PARITY_TOL,
                # ``base_model_id`` is the EXPLICIT provenance the dispatcher's
                # pre-sweep gate reads (concern #3): a 0.5B-smoke-derived decision
                # MUST NOT be accepted for the 7B production sweep. ``model`` kept
                # for back-compat (older salvaged decisions only carry ``model``).
                "base_model_id": gate_c1.get("model"),
                "model": gate_c1.get("model"),
                "note": (
                    "Gate C1.2b decision: use_cache=True when caching does NOT drop the "
                    "patch (parity < tol); use_cache=False runs the sweep uncached as the "
                    "safety net when the patch is dropped or the parity is marginal."
                ),
            },
            indent=2,
            default=float,
        )
    )
    logger.info("[phase=canary] wrote use_cache decision %s (use_cache=%s)", p, use_cache)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
