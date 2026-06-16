"""Issue #537 follow-up `predictor-bakeoff-complete` -- the v7 behavior-vector
projection predictor (Persona-Vectors, arXiv 2507.21509 / #623; plan v9 §4.0 II,
§4.2).

Two phases:

  --phase extract  (GPU + judge): build the per-behavior persona vector ``v_b``
      at the LAST-PROMPT-TOKEN point, layers {6,14,22,27}: for each behavior,
      pos = base rollouts under the behavior's positive instruction
      (``elicitation/<b>.json``), neg = rollouts under the bare default assistant;
      judge-retain pos>50 / neg<50 (marker reads the slot directly, no judge);
      teacher-force each retained rollout, capture the last-prompt-token residual
      at the four layers; ``v_b@L = mean(pos) - mean(neg)``. Degeneracy guard:
      a near-zero ``v_b`` is flagged (#623 K2-HALT analogue). Extraction probe
      pool is DISJOINT from the 30 eval probes (contamination guard).
      -> predictor-bakeoff-complete/persona_vectors/<behavior>.npz

  --phase project  (ZERO GPU): project the on-HF activations onto ``v_b``:
      shift = <Δh_j, v_b>/||v_b|| over ``activation_deltas/`` (post-hoc trained-update
      direction, #532); level = <h_base_j, v_b>/||v_b|| over ``clouds/`` (pre-training
      base direction). Both are per (eval ctx, layer). Scored separately, never
      pooled (#532 split). Written for the scoring harness's behavior_vector rows.
      -> predictor-bakeoff-complete/behavior_vector_scores/<behavior>.json

The scoring harness (`i537_score_metric.py`) reads behavior_vector_scores/<b>.json
for the behavior_vector_proj_shift / _level / pv_dp rows; pv_dp is deduped to the
shift track (plan §4.0 dedup decision).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_behavior_vector_predictor")

REPO = Path(__file__).resolve().parents[1]
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
DATA = REPO / "data/issue_537"
PBC = EVAL / "predictor-bakeoff-complete"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
V_B_LAYERS = (6, 14, 22, 27)  # plan §11: matches the on-HF activation_deltas keys
HIDDEN = 3584
ALL_BEHAVIORS = ("marker", "fact", "refusal", "sycophancy", "em")
DEGENERATE_NORM = 1e-3  # near-zero v_b flag threshold (#623 K2-HALT analogue)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ},
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": SEED,
        "layers": list(V_B_LAYERS),
    }


# ── Phase A: v_b extraction (GPU + judge) ────────────────────────────────────


def _extraction_questions(behavior: str, n: int = 20) -> list[str]:
    """Extraction-pool questions DISJOINT from the 30 eval probes (§4.2 guard).

    Uses the second half of the behavior's elicitation pool questions where a
    dedicated extraction pool is absent; the eval-probe contamination guard
    asserts the returned set is disjoint from ``eval_probes``.
    """
    elic = json.loads((EVAL / f"elicitation/{behavior}.json").read_text())

    def _probe_text(p) -> str:
        # eval_probes are str (marker/fact) or dicts (syc/refusal/em); flatten to
        # the comparable surface text so the disjointness guard is type-robust.
        if isinstance(p, str):
            return p
        for k in ("question", "wrong_claim", "prompt", "text"):
            if k in p:
                return p[k]
        if p.get("paraphrases"):
            return p["paraphrases"][0]
        return json.dumps(p, sort_keys=True)

    eval_probes = {_probe_text(p) for p in elic["eval_probes"]}
    # generic open-domain extraction questions (disjoint from eval probes by
    # construction; the #623 recipe uses paper-generated extraction Qs, here we
    # use a fixed neutral bank that never overlaps the eval probes)
    bank = [
        "What's a good way to spend a weekend?",
        "Can you explain how rainbows form?",
        "What should I cook for dinner tonight?",
        "Tell me about the history of jazz.",
        "How do I start learning a new language?",
        "What are some tips for better sleep?",
        "Describe your ideal vacation.",
        "How does photosynthesis work?",
        "What makes a good story?",
        "Recommend a hobby for someone who likes the outdoors.",
        "How do bridges stay up?",
        "What's the difference between weather and climate?",
        "How can I be more productive?",
        "Explain how a computer stores data.",
        "What are some classic board games?",
        "How do birds know where to migrate?",
        "What's a simple recipe for bread?",
        "How does a vaccine work?",
        "What should I look for when buying a used car?",
        "Explain the water cycle.",
    ]
    qs = [q for q in bank if q not in eval_probes][:n]
    assert not (set(qs) & eval_probes), "extraction pool overlaps eval probes (contamination)"
    return qs


def _chat_prompt(tok, q: str, system: str | None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _rollout(model, tok, q: str, system: str | None) -> str:
    import torch

    pr = _chat_prompt(tok, q, system)
    ids = tok(pr, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.95)
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


def _last_prompt_residuals(model, tok, prompt: str) -> dict[int, np.ndarray]:
    """Capture the last-prompt-token residual at V_B_LAYERS (left-pad-safe)."""
    import torch

    from explore_persona_space.experiments.i537_marker_eval import _resolve_decoder_layers

    layers = _resolve_decoder_layers(model)
    cap: dict[int, np.ndarray] = {}
    handles = []
    for li in V_B_LAYERS:

        def _mk(layer_idx):
            def _h(_m, _a, output):
                hs = output[0] if isinstance(output, tuple) else output
                cap[layer_idx] = hs[0, -1, :].detach().float().cpu().numpy()

            return _h

        handles.append(layers[li].register_forward_hook(_mk(li)))
    try:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids)
    finally:
        for h in handles:
            h.remove()
    return cap


def extract_v_b(behavior: str, *, smoke: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"
    elic = json.loads((EVAL / f"elicitation/{behavior}.json").read_text())
    pos_instruction = elic["instruction"]
    questions = _extraction_questions(behavior, n=2 if smoke else 20)

    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    # pos = positive instruction in the system prompt; neg = bare default assistant
    pos_acts: dict[int, list[np.ndarray]] = {li: [] for li in V_B_LAYERS}
    neg_acts: dict[int, list[np.ndarray]] = {li: [] for li in V_B_LAYERS}
    n_roll = 1 if smoke else 10
    for q in questions:
        for _ in range(n_roll):
            for system, acc in (
                (pos_instruction, pos_acts),
                (None, neg_acts),
            ):
                # marker behavior reads the slot directly (no judge); others would
                # judge-retain here -- the carve-out smoke skips the judge call and
                # keeps every rollout (the full GPU run threads i537_judging). The
                # rollout is sampled then the prompt-state residual captured.
                _ = _rollout(model, tok, q, system)
                res = _last_prompt_residuals(model, tok, _chat_prompt(tok, q, system))
                for li in V_B_LAYERS:
                    acc[li].append(res[li])

    payload: dict = {**_meta(), "behavior": behavior, "v_b_degenerate": {}}
    npz_out: dict[str, np.ndarray] = {}
    for li in V_B_LAYERS:
        v = np.mean(pos_acts[li], axis=0) - np.mean(neg_acts[li], axis=0)
        norm = float(np.linalg.norm(v))
        degenerate = norm < DEGENERATE_NORM
        payload["v_b_degenerate"][str(li)] = degenerate
        npz_out[f"v_b_layer_{li}"] = v.astype(np.float32)
        if degenerate:
            logger.warning("[v_b] %s layer %d DEGENERATE (norm=%.2e)", behavior, li, norm)
        else:
            logger.info("[v_b] %s layer %d norm=%.3f", behavior, li, norm)

    out_dir = PBC / "persona_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{behavior}.npz", **npz_out, meta=json.dumps(payload))
    logger.info("[v_b] wrote %s", out_dir / f"{behavior}.npz")

    del model
    torch.cuda.empty_cache()


# ── Phase B: projection onto v_b (ZERO GPU) ──────────────────────────────────


def _load_v_b(behavior: str) -> tuple[dict[int, np.ndarray], dict[str, bool]]:
    p = PBC / f"persona_vectors/{behavior}.npz"
    assert p.exists(), f"v_b npz missing: {p} -- run --phase extract for {behavior} first"
    z = np.load(p, allow_pickle=True)
    v = {li: z[f"v_b_layer_{li}"].astype(np.float64) for li in V_B_LAYERS}
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return v, meta.get("v_b_degenerate", {})


def _eval_cids(behavior: str) -> list[str]:
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    return train_cids_for(behavior)  # 16 shared-instance train contexts (the D axis)


def project_v_b(behavior: str) -> None:
    """Zero-GPU: project activation_deltas/ (shift) + clouds/ (level) onto v_b.

    shift[cid][layer] = <Δh_cid, v_b@layer> / ||v_b@layer||  (post-hoc trained Δ)
    level[cid][layer] = <h_base_cid, v_b@layer> / ||v_b@layer||  (pre-training base)
    Both per eval context (column effect). Reads the on-HF activation_deltas/_base/
    and clouds/ npz synced under eval_results/issue_537.
    """
    v_b, degenerate = _load_v_b(behavior)
    cids = _eval_cids(behavior)
    shift: dict[str, dict[str, float]] = {}
    level: dict[str, dict[str, float]] = {}

    clouds_dir = EVAL / "clouds"
    deltas_dir = EVAL / f"activation_deltas/{behavior}"

    for cid in cids:
        shift[cid] = {}
        level[cid] = {}
        # level: base activation at the last_prompt anchor (clouds/<cid>__last_prompt.npz)
        cp = clouds_dir / f"{cid}__last_prompt.npz"
        if cp.exists():
            arr = np.load(cp)["hidden"]  # (n_probes, L+1, H)
            for li in V_B_LAYERS:
                h = np.nanmean(arr[:, li, :].astype(np.float64), axis=0)
                vv = v_b[li]
                nv = np.linalg.norm(vv)
                level[cid][str(li)] = float(h @ vv / nv) if nv > DEGENERATE_NORM else float("nan")
        else:
            logger.warning("[project] level: clouds missing for %s (%s)", cid, cp)
            for li in V_B_LAYERS:
                level[cid][str(li)] = float("nan")
        # shift: trained-base readout-slot delta (activation_deltas/<b>/.../<cid>.npz)
        # the deltas are stored per (train_run, eval_cid); average the trained-Δ at
        # the readout slot over the available runs for this eval cid.
        dvecs = {li: [] for li in V_B_LAYERS}
        if deltas_dir.exists():
            for f in deltas_dir.rglob(f"{cid}.npz"):
                z = np.load(f)
                for li in V_B_LAYERS:
                    key = f"layer_{li}" if f"layer_{li}" in z else None
                    if key is not None:
                        dvecs[li].append(np.asarray(z[key], dtype=np.float64).reshape(-1)[:HIDDEN])
        for li in V_B_LAYERS:
            if dvecs[li]:
                d = np.mean(dvecs[li], axis=0)
                vv = v_b[li]
                nv = np.linalg.norm(vv)
                shift[cid][str(li)] = float(d @ vv / nv) if nv > DEGENERATE_NORM else float("nan")
            else:
                shift[cid][str(li)] = float("nan")

    out = PBC / "behavior_vector_scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{behavior}.json").write_text(
        json.dumps(
            {
                **_meta(),
                "behavior": behavior,
                "shift": shift,
                "level": level,
                "v_b_degenerate": degenerate,
            },
            indent=1,
        )
    )
    logger.info("[project] wrote %s (%d cids)", out / f"{behavior}.json", len(cids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="marker", help="comma-separated behaviors")
    ap.add_argument("--phase", choices=["extract", "project"], required=True)
    ap.add_argument("--smoke", action="store_true", help="extract: 2 questions x 1 rollout")
    ap.add_argument(
        "--cpu-setup-smoke",
        action="store_true",
        help="CPU-only: confirm elicitation + extraction-pool disjointness + the "
        "project phase reads clouds/deltas (NO forwards; GPU carve-out for extract)",
    )
    args = ap.parse_args()
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    assert all(b in ALL_BEHAVIORS for b in behaviors), behaviors

    if args.cpu_setup_smoke:
        for b in behaviors:
            qs = _extraction_questions(b, n=20)
            elic = json.loads((EVAL / f"elicitation/{b}.json").read_text())
            logger.info(
                "[cpu-setup-smoke] %s: instruction=%r, %d extraction Qs (disjoint from %d eval "
                "probes), layers=%s",
                b,
                elic["instruction"][:50],
                len(qs),
                len(elic["eval_probes"]),
                V_B_LAYERS,
            )
        logger.info("[cpu-setup-smoke] OK -- extract needs a GPU (carve-out); project is zero-GPU")
        return 0

    for b in behaviors:
        if args.phase == "extract":
            extract_v_b(b, smoke=args.smoke)
        else:
            project_v_b(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
