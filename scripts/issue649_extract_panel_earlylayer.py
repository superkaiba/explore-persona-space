#!/usr/bin/env python3
# ruff: noqa: RUF002
# (multiplication-sign / arrow characters intentional in docstrings/labels)
"""Issue #649 Phase-1 early-layer geometry extractor (CPU-only, no pod).

Re-extracts the #509 early-layer-band residual-stream geometry on the #612
sycophancy panel, for the LEVEL/CHANGE predictor decomposition (#649). #509
established that for SYCOPHANCY the geometry signal lives at the early-layer band
(end-of-system L2 / last-prompt L7), NOT the late-layer L20 that won on the
marker — so the on-HF ``panel_centroids_layer20.pt`` is the WEAK cell and we
extract the early band fresh.

Per persona (30 #612 panel personas + the 4 sources — the sources ARE in the
panel, so 34 = 30 unique here; see ``_resolve_personas``), forward the base model
``Qwen/Qwen2.5-7B-Instruct`` over a fixed persona-neutral probe bank under the
persona's system prompt, capturing residual activations at the two #509 cells:

  (1) ``end_of_system`` × L2  — residual at the last token of the system-only
      prefix (input-independent → ONE centroid vector per persona). The PRIMARY
      cosine cell.
  (2) ``last_prompt``    × L7 — residual at the last input token after a user
      turn (varies by probe → a per-persona CLOUD). The SECONDARY cosine cell
      AND the source of the Gaussian-KL clouds.

We ALSO capture ``last_prompt`` clouds at L2 (so the Gaussian-KL "early band" has
a cloud at the primary layer too — ``end_of_system`` is a single vector and has
no cloud for KL).

Outputs (``data/issue_649/inputs/early_layer_geometry.npz`` by default):
  - ``eos_L2_centroid`` (P, H)   — end_of_system L2 centroid per persona (cosine primary)
  - ``lastprompt_L2_cloud`` (P, n_probes, H) — last_prompt L2 cloud (KL@L2)
  - ``lastprompt_L7_cloud`` (P, n_probes, H) — last_prompt L7 cloud (cosine secondary + KL @ L7)
  - persona_names, probe_sha256, layers, n_probes, centering, repro metadata

``--smoke``: a persona subset (default villain + first 3 bystanders) × the SAME
probe bank through the SAME code path, into ``early_layer_geometry_smoke.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i649_extract_panel_earlylayer")

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# The #509 early-layer band cells: end_of_system L2 (primary cosine),
# last_prompt L7 (secondary cosine), L2 cloud carried for KL @ the primary layer.
LAYER_EOS_PRIMARY = 2
LAYER_LASTPROMPT_SECONDARY = 7
LASTPROMPT_LAYERS = (2, 7)  # clouds captured at both early-band layers
GKL_K = 16  # the #502 bakeoff winner (via #532); KL needs >= 2k probes per persona
N_PROBES_FULL = 40  # >= 2*k=32 so the k=16 PCA-subspace covariance is non-singular
N_PROBES_SMOKE = 40  # smoke uses the SAME probe bank (still >= 32; only personas shrink)
DATA_REPO = "superkaiba1/explore-persona-space-data"
PANEL_SET_REPO_PATH = "issue612_sycophancy_onpolicy/panel/panel_set.json"
# The 4 #612 sources (the keys of each panel persona's `cosines` dict).
SOURCES_CANNED = ("villain", "comedian", "kindergarten_teacher", "software_engineer")
SOURCES_ONPOLICY = ("villain", "comedian")

# Persona-neutral probe bank — 40 generic user turns. We deliberately use
# topic-neutral probes (NOT #444's courthouse entity) so the per-persona cloud
# is driven by the PERSONA DIRECTION (the system prompt), not by probe topic.
# Fixed + committed in-script => deterministic, reproducible, self-contained.
PROBE_BANK: tuple[str, ...] = (
    "What do you think about the weather today?",
    "Can you help me plan my weekend?",
    "Tell me something interesting.",
    "What is your opinion on modern art?",
    "How should I approach learning a new skill?",
    "What makes a good story?",
    "Can you recommend a book to read?",
    "What is the best way to stay organized?",
    "How do you feel about technology?",
    "What advice would you give to a student?",
    "Describe an ideal day.",
    "What is your favorite kind of music?",
    "How can I be more productive?",
    "What do you think about traveling?",
    "Can you explain how to cook a simple meal?",
    "What is the meaning of a good conversation?",
    "How do I make a difficult decision?",
    "What is something worth learning?",
    "Tell me about a place you find interesting.",
    "What are some good habits to build?",
    "How should I spend my free time?",
    "What do you think makes people happy?",
    "Can you help me write a short message?",
    "What is your take on working from home?",
    "How can I improve my writing?",
    "What is a good way to start the morning?",
    "Tell me about something you find beautiful.",
    "How do you handle a busy schedule?",
    "What is the value of asking questions?",
    "Can you suggest a hobby to try?",
    "What do you think about giving advice?",
    "How can I be a better listener?",
    "What is your view on taking risks?",
    "Describe a useful piece of advice.",
    "How should I prepare for a presentation?",
    "What makes a community strong?",
    "Can you help me think through a problem?",
    "What do you think about change?",
    "How do I stay motivated over time?",
    "What is something you would recommend trying?",
)
assert len(PROBE_BANK) >= 2 * GKL_K, (
    f"probe bank too small for k={GKL_K}: need >= {2 * GKL_K}, have {len(PROBE_BANK)}"
)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _git_commit_sha() -> str:
    import os
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},  # explicit per the subprocess-env rule (no implicit inherit)
        check=False,
    )
    return out.stdout.strip() or "unknown"


def _repro_metadata() -> dict[str, Any]:
    import platform

    import torch
    import transformers

    return {
        "git_commit": _git_commit_sha(),
        "base_model": BASE_MODEL,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "timestamp": _now_iso(),
    }


def _probe_sha256(probes: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(probes), ensure_ascii=False).encode()).hexdigest()


def _load_panel_set(inputs_dir: Path) -> dict:
    """Download the #612 panel_set.json into the issue-owned inputs dir, pinned
    to the data-repo head SHA. Returns the parsed dict (30-persona panel)."""
    from huggingface_hub import hf_hub_download

    local = inputs_dir / "panel_set.json"
    if not local.exists():
        inputs_dir.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(DATA_REPO, PANEL_SET_REPO_PATH, repo_type="dataset", revision="main")
        local.write_bytes(Path(src).read_bytes())
        logger.info("downloaded panel_set -> %s", local)
    ps = json.loads(local.read_text())
    assert "personas" in ps and len(ps["personas"]) == 30, (
        f"expected 30 panel personas, got {len(ps.get('personas', {}))}"
    )
    return ps


def _resolve_personas(panel_set: dict, smoke: bool) -> tuple[list[str], dict[str, str]]:
    """Return (persona_names, name->system_prompt). The 4 sources are already
    members of the 30-persona panel, so the bank is exactly the 30 panel
    personas (NOT 30+4=34 distinct vectors). Smoke: villain + first 3
    non-source bystanders."""
    personas = panel_set["personas"]
    pool = {name: personas[name]["prompt"] for name in personas}
    # every source must be present in the panel (sanity)
    for s in SOURCES_CANNED:
        assert s in pool, f"source {s!r} missing from panel"
    if smoke:
        # villain (a source) + the first 6 panel personas that are NOT the
        # smoke source — a tiny, deterministic subset through the same path.
        # 6 non-villain bystanders so the decomp smoke clears _cv_r2_grouped's
        # len(y)<5 floor and the bystander-grouped M0 identifiability check
        # actually reads a non-NaN CV-R² (the plan's smoke assertion).
        names = ["villain"]
        for n in personas:
            if n != "villain" and len(names) < 7:
                names.append(n)
        return names, {n: pool[n] for n in names}
    return list(personas.keys()), pool


# ---------------------------------------------------------------------------
# Activation extraction (forward hooks on model.model.layers[L]; the #404/#493
# canonical handle — captures the pre-final-norm block output uniformly).
# ---------------------------------------------------------------------------
def _extract_persona(
    model,
    tokenizer,
    system_prompt: str,
    probes: tuple[str, ...],
    device: str,
) -> dict[str, np.ndarray]:
    """For one persona return:
      - 'eos_L2': (H,)               end_of_system L2 centroid (one fwd, input-independent)
      - 'lp_L2':  (n_probes, H)      last_prompt L2 cloud
      - 'lp_L7':  (n_probes, H)      last_prompt L7 cloud
    via forward hooks. All fp32 numpy on CPU."""
    import torch

    layers = (LAYER_EOS_PRIMARY, LAYER_LASTPROMPT_SECONDARY)
    captures: dict[int, list] = {li: [] for li in layers}

    def make_hook(li):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[li].append(hs.detach())

        return hook_fn

    handles = []
    for li in layers:
        if len(model.model.layers) <= li:
            raise IndexError(f"layer {li} out of range; model has {len(model.model.layers)} blocks")
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    try:
        # ── end_of_system: forward the system-only prefix once ──
        system_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tokenizer(system_text, return_tensors="pt", add_special_tokens=False).to(device)
        for li in layers:
            captures[li].clear()
        with torch.no_grad():
            _ = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
        sys_last = ids["input_ids"].shape[1] - 1
        hs2 = captures[LAYER_EOS_PRIMARY][-1]
        assert hs2.shape[0] == 1, hs2.shape
        eos_L2 = hs2[0, sys_last, :].float().cpu().numpy().astype(np.float32)

        # ── last_prompt: forward (system + each probe), capture last-input token ──
        lp_L2: list[np.ndarray] = []
        lp_L7: list[np.ndarray] = []
        for q in probes:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            pids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            for li in layers:
                captures[li].clear()
            with torch.no_grad():
                _ = model(input_ids=pids["input_ids"], attention_mask=pids["attention_mask"])
            last = pids["input_ids"].shape[1] - 1
            lp_L2.append(captures[2][-1][0, last, :].float().cpu().numpy().astype(np.float32))
            lp_L7.append(captures[7][-1][0, last, :].float().cpu().numpy().astype(np.float32))
        return {
            "eos_L2": eos_L2,
            "lp_L2": np.stack(lp_L2),
            "lp_L7": np.stack(lp_L7),
        }
    finally:
        for h in handles:
            h.remove()


def extract_all(
    persona_names: list[str],
    pool: dict[str, str],
    probes: tuple[str, ...],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {persona: {'eos_L2', 'lp_L2', 'lp_L7'}}. Loads the base model once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s (bf16) on %s for %d personas", BASE_MODEL, device, len(persona_names))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map=device
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    out: dict[str, dict[str, np.ndarray]] = {}
    n_probes = len(probes)
    for i, name in enumerate(persona_names):
        t0 = time.time()
        acts = _extract_persona(model, tok, pool[name], probes, device)
        # n_probes assert (Risk-row-5 / Assumption 7): >= 2*k for non-singular k=16 cov
        assert acts["lp_L2"].shape[0] == n_probes >= 2 * GKL_K, (name, acts["lp_L2"].shape)
        assert acts["lp_L7"].shape[0] == n_probes, (name, acts["lp_L7"].shape)
        out[name] = acts
        logger.info(
            "extracted %s (%d/%d) in %.1fs", name, i + 1, len(persona_names), time.time() - t0
        )
    del model
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="persona subset, smoke namespace")
    ap.add_argument("--gpu-id", type=int, default=None, help="pin CUDA_VISIBLE_DEVICES")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .npz (default data/issue_649/inputs/early_layer_geometry[_smoke].npz)",
    )
    ap.add_argument(
        "--inputs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_649" / "inputs",
        help="issue-owned inputs dir for the pinned panel_set.json",
    )
    args = ap.parse_args()

    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # before torch import
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_path = args.out or (
        args.inputs_dir
        / ("early_layer_geometry_smoke.npz" if args.smoke else "early_layer_geometry.npz")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_set = _load_panel_set(args.inputs_dir)
    persona_names, pool = _resolve_personas(panel_set, args.smoke)
    probes = PROBE_BANK  # same bank for smoke + full (>= 32 either way)
    probe_sha = _probe_sha256(probes)
    logger.info(
        "personas=%d probes=%d (sha %s) device=%s eos_L%d/lp_L%d",
        len(persona_names),
        len(probes),
        probe_sha[:12],
        device,
        LAYER_EOS_PRIMARY,
        LAYER_LASTPROMPT_SECONDARY,
    )

    t0 = time.time()
    acts = extract_all(persona_names, pool, probes, device)
    extract_s = time.time() - t0

    H = acts[persona_names[0]]["eos_L2"].shape[0]
    eos_L2_centroid = np.stack([acts[n]["eos_L2"] for n in persona_names])  # (P, H)
    lp_L2_cloud = np.stack([acts[n]["lp_L2"] for n in persona_names])  # (P, n_probes, H)
    lp_L7_cloud = np.stack([acts[n]["lp_L7"] for n in persona_names])  # (P, n_probes, H)
    assert eos_L2_centroid.shape == (len(persona_names), H), eos_L2_centroid.shape
    assert lp_L7_cloud.shape == (len(persona_names), len(probes), H), lp_L7_cloud.shape

    meta = {
        "_doc": (
            "Issue #649 early-layer geometry on the #612 sycophancy panel. "
            "eos_L2_centroid: end_of_system L2 (PRIMARY cosine cell, input-independent). "
            "lastprompt_L7_cloud: last_prompt L7 (SECONDARY cosine = cloud mean; KL @ L7). "
            "lastprompt_L2_cloud: last_prompt L2 cloud (KL @ the primary layer). "
            "Centered bank cosine is computed downstream (centering=global_mean)."
        ),
        "model": BASE_MODEL,
        "persona_names": list(persona_names),
        "sources_canned": list(SOURCES_CANNED),
        "sources_onpolicy": list(SOURCES_ONPOLICY),
        "n_probes": len(probes),
        "probe_sha256": probe_sha,
        "gkl_k": GKL_K,
        "layer_eos_primary": LAYER_EOS_PRIMARY,
        "layer_lastprompt_secondary": LAYER_LASTPROMPT_SECONDARY,
        "lastprompt_layers": list(LASTPROMPT_LAYERS),
        "centering": "global_mean",
        "smoke": args.smoke,
        "hidden_dim": int(H),
        "wall_seconds_extract": round(extract_s, 1),
        "reproducibility": _repro_metadata(),
    }
    np.savez(
        out_path,
        eos_L2_centroid=eos_L2_centroid.astype(np.float32),
        lastprompt_L2_cloud=lp_L2_cloud.astype(np.float32),
        lastprompt_L7_cloud=lp_L7_cloud.astype(np.float32),
        meta_json=np.array(json.dumps(meta, default=str)),
    )
    logger.info(
        "WROTE %s (%d personas x %d probes, H=%d) in %.1fs",
        out_path,
        len(persona_names),
        len(probes),
        H,
        extract_s,
    )


if __name__ == "__main__":
    main()
