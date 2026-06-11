# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, σ, γ, ※, —) in scientific docstrings + labels.
"""Task #604 Phase B — base-model context-vector extraction (pod, 1×H100).

Extracts, for ~50–55 contexts × 50 probes, the last-prompt-token state at
THREE capture points per layer, in ONE forward per prompt (plan §4 Phase B):

(i)   attn module input  = OUTPUT of ``layers[l].input_layernorm``
      (RMSNorm + γ applied PER PROBE — exactly what q/k/v read);
(ii)  MLP module input   = OUTPUT of ``layers[l].post_attention_layernorm``
      (exactly what gate/up read);
(iii) raw pre-block residual = the block INPUT (sensitivity + write reads).

Per-context centroids for (i)/(ii) are means over probes computed AFTER the
per-probe normalization (round-1 methodology binding fix). Persists
per-context shards as each context completes (checkpoint-per-phase), then
assembles the bundle:

- ``context_vectors_all_layers.pt``  raw-residual centroids (fp32)
- ``module_input_centroids.pt``      TRUE module-input centroids (fp32)
- ``per_probe_vectors_fp16.pt``      all three capture points, per probe
- ``rmsnorm_gamma.pt``               input/post-attention LN γ, all layers
- ``dispersion_diagnostics.json``    split-half centroid cos + probe spread
- ``manifest.json``                  contexts, groups, prompt text + sha256

``--upload`` pushes the bundle to the HF data repo
``issue604_adapter_svd/analysis_tensors/`` in ONE commit (fail-loud) before
the pod terminates (upload-policy: plan-referenced analysis tensors MUST
land before termination).

Smoke = this same entrypoint with ``--context-names ... --probes 2`` AND a
SEPARATE ``--out-dir`` (e.g. ``eval_results/issue_604/context_vectors_smoke``)
so production runs can never consume smoke artifacts. Every shard embeds its
run parameters (model, n_probes, probes_sha256, dtype, prompt hash); resume
VALIDATES existing shards against the current run before skipping and fails
loud on any mismatch (``--force`` recomputes in place). Optionally
``--model Qwen/Qwen2.5-0.5B-Instruct`` for a CPU-budget run; dims are read
from the model config, the 28/3584 assert applies only to the production
model.

Pod-side contract: emits ``[phase=...]`` lines ending in ``[phase=done]``
and writes the poll_pipeline results sentinel when ``/workspace`` exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)
from explore_persona_space.experiments.issue_604 import (  # noqa: E402
    AUGMENT_PERSONAS_311,
    BASE_MODEL,
    HF_BUCKET,
    HF_DATA_REPO,
    HIDDEN_SIZE,
    I474_SOURCES,
    I518_SOURCE_PROMPTS,
    I519_NEGATIVES,
    I519_SOURCE,
    I551_PANEL_14,
    N_LAYERS,
    extract_mix_prompts,
    load_persona_bank,
    result_metadata,
    sha256_text,
)

logger = logging.getLogger("issue604.phase_b")

# The dial eval panel (row order of the §5 shift tensors; asserted against a
# sibling shift JSON when available in the checkout).
DIAL_EVAL_PANEL_19 = (
    "paramedic",
    "surgeon",
    "poet",
    "navy_seal",
    "army_medic",
    "florist",
    "cybersec_consultant",
    "pentester",
    "private_investigator",
    "librarian",
    "software_engineer",
    "data_scientist",
    "medical_doctor",
    "kindergarten_teacher",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "assistant",
)

DIAL_MIXES_FOR_PROMPTS = (
    "issue_538/training_mixes/florist__medical_doctor__joint__seed42.jsonl",
    "issue_538/training_mixes/librarian__police_officer__joint__seed42.jsonl",
)
I519_MIX_FOR_PROMPTS = "issue_519/marker_seed42.jsonl"


def _persona_messages(system_prompt: str | None, question: str, tokenizer) -> str:
    """Chat-template prompt (system persona + user q, generation prompt)."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def assemble_contexts(tokenizer, probes: list[str]) -> dict[str, dict]:
    """Build the ~50–55 context union (plan §4 Phase B table).

    Returns {context_name: {"group", "system_prompt", "prompts": [str, ...],
    "prompt_sha256"}}. Name collisions across groups: byte-identical
    payloads dedup (group membership recorded); DIFFERENT payloads register
    under ``<name>__<group>`` so no context silently shadows another.
    """
    from huggingface_hub import hf_hub_download

    contexts: dict[str, dict] = {}

    def _add(name: str, group: str, system_prompt: str | None, prompts: list[str]) -> None:
        key = "\x00".join(prompts)
        if name in contexts:
            if contexts[name]["_key"] == key:
                contexts[name]["groups"].append(group)
                return
            name = f"{name}__{group}"
            assert name not in contexts, f"double collision for {name}"
        contexts[name] = {
            "groups": [group],
            "system_prompt": system_prompt,
            "prompts": prompts,
            "prompt_sha256": sha256_text(key),
            "_key": key,
        }

    # 1) #474 transformations (16) — the #560 context construction, verbatim.
    class_d = load_class_d_rewrites()
    for cid in I474_SOURCES:
        prompts = [
            build_prompt_for_condition(
                CONDITIONS_BY_ID[cid], q, tokenizer, class_d_rewrites=class_d
            )
            for q in probes
        ]
        _add(cid, "i474_transformations", CONDITIONS_BY_ID[cid].system_prompt, prompts)

    # 2) Marker-dial sources + negatives (8) — EXACT trained system prompts
    #    from the #538 training mixes (HF, verified).
    mix_prompts: dict[str, str | None] = {}
    for rel in DIAL_MIXES_FOR_PROMPTS:
        path = Path(hf_hub_download(HF_DATA_REPO, rel, repo_type="dataset"))
        for name, sp in extract_mix_prompts(path).items():
            if name in mix_prompts:
                assert mix_prompts[name] == sp, f"mix prompt drift for {name!r} across mixes"
            mix_prompts[name] = sp
    expected_dial = {
        "florist",
        "medical_doctor",
        "librarian",
        "police_officer",
        "assistant",
        "programmer",
        "chef",
        "kindergarten_teacher",
    }
    missing = expected_dial - set(mix_prompts)
    assert not missing, f"dial mixes did not cover personas: {sorted(missing)}"
    bank = load_persona_bank(PROJECT_ROOT)
    for name in sorted(expected_dial):
        sp = mix_prompts[name]
        if name in bank and sp is not None and bank[name] != sp:
            logger.warning(
                "trained prompt differs from bank for %r — using the TRAINED prompt", name
            )
        _add(name, "dial_trained", sp, [_persona_messages(sp, q, tokenizer) for q in probes])

    # 3) Dial eval panel (19) — persona bank + the vendored #311 augments.
    for name in DIAL_EVAL_PANEL_19:
        sp = bank.get(name) or AUGMENT_PERSONAS_311.get(name)
        assert sp is not None, f"eval-panel persona {name!r} resolves in neither bank nor augments"
        _add(name, "dial_eval_panel", sp, [_persona_messages(sp, q, tokenizer) for q in probes])

    # 4) #519 source + trained negatives (exact mix prompts) + #551 panel.
    path = Path(hf_hub_download(HF_DATA_REPO, I519_MIX_FOR_PROMPTS, repo_type="dataset"))
    i519_prompts = extract_mix_prompts(path)
    expected_519 = {I519_SOURCE, *I519_NEGATIVES}
    missing = expected_519 - set(i519_prompts)
    assert not missing, f"#519 mix did not cover personas: {sorted(missing)}"
    for name in sorted(expected_519):
        sp = i519_prompts[name]
        _add(name, "i519_trained", sp, [_persona_messages(sp, q, tokenizer) for q in probes])
    from explore_persona_space.personas import PERSONAS as MAIN_PERSONAS

    for name in I551_PANEL_14:
        sp = bank.get(name) or AUGMENT_PERSONAS_311.get(name) or MAIN_PERSONAS.get(name)
        assert sp is not None, f"#551 panel persona {name!r} unresolvable (bank/augments/PERSONAS)"
        _add(name, "i551_panel", sp, [_persona_messages(sp, q, tokenizer) for q in probes])

    # 5) #518 sources (6) — vendored verbatim from the issue-518 branch.
    # ``_add`` dedups byte-identical payloads and renames on conflict
    # (e.g. a #518 prompt that differs from the bank's same-named persona
    # lands as ``<name>__i518_sources``).
    for name, sp in I518_SOURCE_PROMPTS.items():
        _add(name, "i518_sources", sp, [_persona_messages(sp, q, tokenizer) for q in probes])

    for ctx in contexts.values():
        del ctx["_key"]
    return contexts


def extract_all(
    model,
    tokenizer,
    contexts: dict[str, dict],
    shard_dir: Path,
    shard_meta: dict,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """One forward per prompt; three capture points per layer; shard per context.

    ``shard_meta`` (model, n_probes, probes_sha256, dtype) is embedded in
    every shard so a later resume can validate compatibility before skipping
    (stale-cache guard); ``force=True`` recomputes existing shards in place.
    Returns (n_layers, hidden) read from the captures.
    """
    import torch

    layers = model.model.layers
    n_layers = len(layers)
    captured: dict[tuple[str, int], torch.Tensor] = {}

    def _ln_hook(kind: str, layer_idx: int):
        def hook(module, inputs, output):
            captured[(kind, layer_idx)] = output.detach()

        return hook

    def _pre_hook(layer_idx: int):
        def hook(module, args, kwargs):
            # hidden_states is positional in current transformers; tolerate a
            # kwargs-only call shape (fail loud if neither carries it).
            hs = args[0] if args else kwargs["hidden_states"]
            captured[("raw", layer_idx)] = hs.detach()

        return hook

    handles = []
    for li, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_hook(_ln_hook("attn", li)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_ln_hook("mlp", li)))
        handles.append(layer.register_forward_pre_hook(_pre_hook(li), with_kwargs=True))

    hidden = model.config.hidden_size
    try:
        for ci, (name, ctx) in enumerate(contexts.items()):
            shard_path = shard_dir / f"{name}.pt"
            if shard_path.exists() and not force:
                # Compatibility was validated by the pre-pass in main()
                # BEFORE model load (stale-cache guard).
                logger.info("[%d/%d] skip (shard exists): %s", ci + 1, len(contexts), name)
                continue
            n_p = len(ctx["prompts"])
            per_probe = {
                kind: np.zeros((n_p, n_layers, hidden), dtype=np.float16)
                for kind in ("attn", "mlp", "raw")
            }
            # Centroids accumulate in float64 from the fp32 captures — the
            # fp16 per-probe arrays are SIDECARS only, never centroid inputs.
            sums = {
                kind: np.zeros((n_layers, hidden), dtype=np.float64)
                for kind in ("attn", "mlp", "raw")
            }
            for pi, text in enumerate(ctx["prompts"]):
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    model(**inputs)
                last = inputs["input_ids"].shape[1] - 1
                for kind in ("attn", "mlp", "raw"):
                    for li in range(n_layers):
                        vec = captured[(kind, li)][0, last, :].float().cpu().numpy()
                        assert vec.shape == (hidden,), vec.shape
                        assert np.isfinite(vec).all(), (name, kind, li, "non-finite capture")
                        v16 = vec.astype(np.float16)
                        assert np.isfinite(v16).all(), (
                            name,
                            kind,
                            li,
                            "fp16 overflow in per-probe sidecar (|x| > 65504)",
                        )
                        per_probe[kind][pi, li, :] = v16
                        sums[kind][li, :] += vec.astype(np.float64)
                captured.clear()
            shard = {
                "name": name,
                "groups": ctx["groups"],
                "meta": {
                    **shard_meta,
                    "n_layers": n_layers,
                    "hidden": hidden,
                    "prompt_sha256": ctx["prompt_sha256"],
                },
                "per_probe_fp16": {k: torch.from_numpy(v) for k, v in per_probe.items()},
                "centroid_raw": torch.from_numpy((sums["raw"] / n_p).astype(np.float32)),
                "centroid_attn": torch.from_numpy((sums["attn"] / n_p).astype(np.float32)),
                "centroid_mlp": torch.from_numpy((sums["mlp"] / n_p).astype(np.float32)),
            }
            torch.save(shard, shard_path)
            logger.info("[%d/%d] context %s done", ci + 1, len(contexts), name)
    finally:
        for h in handles:
            h.remove()
    return n_layers, hidden


def _dispersion(per_probe: np.ndarray) -> dict:
    """Split-half centroid cos, mean pairwise probe cos, variance trace per layer."""
    p, n_layers, _h = per_probe.shape
    arr = per_probe.astype(np.float32)
    out = {"split_half_cos": [], "mean_pairwise_cos": [], "variance_trace": []}
    half = p // 2
    for li in range(n_layers):
        X = arr[:, li, :]
        a, b = X[:half].mean(0), X[half:].mean(0)
        out["split_half_cos"].append(
            float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        )
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-30, None)
        G = Xn @ Xn.T
        off = G[np.triu_indices(p, k=1)]
        out["mean_pairwise_cos"].append(float(off.mean()) if off.size else 1.0)
        out["variance_trace"].append(float(X.var(axis=0).sum()))
    return out


def _validate_existing_shards(shard_dir: Path, contexts: dict[str, dict], run_meta: dict) -> None:
    """Stale-shard validation pre-pass (BEFORE model load).

    A resume may only skip a shard produced under the SAME run parameters;
    anything else (a smoke shard under a production run, a pre-meta shard, a
    different probe set / model / dtype / prompt hash) fails loud here.
    """
    import torch

    stale: list[str] = []
    for name, ctx in contexts.items():
        sp = shard_dir / f"{name}.pt"
        if not sp.exists():
            continue
        smeta = torch.load(sp, weights_only=True).get("meta") or {}
        expected = {**run_meta, "prompt_sha256": ctx["prompt_sha256"]}
        if not smeta:
            bad = ["meta MISSING (pre-validation shard schema — treat as stale)"]
        else:
            bad = [
                f"{k}: shard={smeta.get(k)!r} != run={v!r}"
                for k, v in expected.items()
                if smeta.get(k) != v
            ]
        if bad:
            stale.append(f"{sp}: " + "; ".join(bad))
    if stale:
        raise RuntimeError(
            "stale Phase-B shard(s) would be silently reused — refusing to resume:\n  "
            + "\n  ".join(stale)
            + "\nFix: use a SEPARATE --out-dir for smoke runs "
            "(e.g. eval_results/issue_604/context_vectors_smoke), pass --force to "
            "recompute in place, or delete the stale shards."
        )


def main() -> None:
    """Phase B entrypoint — same code path for smoke and production."""
    parser = argparse.ArgumentParser(
        description="Task 604 Phase B: all-layer context-vector extraction on the base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--probes", type=int, default=50)
    parser.add_argument("--contexts", type=int, default=0, help="restrict to first N (0 = all)")
    parser.add_argument(
        "--context-names",
        default="",
        help="comma-separated explicit context subset (smoke; same loop as the full set)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="float32 for the CPU smoke (bf16 CPU matmul is slow); production stays bf16",
    )
    parser.add_argument("--layers", default="all", help="accepted for CLI parity; always all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute existing shards in place (skips the stale-shard validation pre-pass)",
    )
    parser.add_argument("--upload", action="store_true", help="push bundle to the HF data repo")
    parser.add_argument(
        "--upload-repo", default=HF_DATA_REPO, help="override target repo (quota recovery)"
    )
    parser.add_argument(
        "--out-dir", default=str(PROJECT_ROOT / "eval_results/issue_604/context_vectors")
    )
    args = parser.parse_args()
    assert args.layers == "all", "the key exists per layer — extraction is always all-layer"
    assert args.probes >= 2, "need >= 2 probes (split-half dispersion diagnostics)"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("[phase=b_assemble]", flush=True)
    t0 = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    probes = load_q_test_extended_50()[: args.probes]
    assert len(probes) == args.probes, (len(probes), args.probes)

    contexts = assemble_contexts(tokenizer, probes)
    if args.context_names:
        wanted = [n.strip() for n in args.context_names.split(",") if n.strip()]
        missing = [n for n in wanted if n not in contexts]
        assert not missing, f"--context-names not in the assembled union: {missing}"
        contexts = {n: contexts[n] for n in wanted}
    if args.contexts > 0:
        names = list(contexts)[: args.contexts]
        contexts = {n: contexts[n] for n in names}
    logger.info("assembled %d contexts * %d probes", len(contexts), len(probes))
    # Eyeball check (plan assumptions 10/16): one prompt per group.
    seen_groups: set[str] = set()
    for name, ctx in contexts.items():
        g = ctx["groups"][0]
        if g not in seen_groups:
            seen_groups.add(g)
            logger.info("eyeball [%s] %s system_prompt=%r", g, name, ctx["system_prompt"])

    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    probes_sha = sha256_text("\x00".join(probes))
    run_meta = {
        "model": args.model,
        "n_probes": len(probes),
        "probes_sha256": probes_sha,
        "dtype": args.dtype,
    }
    if not args.force:
        _validate_existing_shards(shard_dir, contexts, run_meta)

    print("[phase=b_extract]", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    if args.model == BASE_MODEL:
        assert model.config.num_hidden_layers == N_LAYERS, model.config.num_hidden_layers
        assert model.config.hidden_size == HIDDEN_SIZE, model.config.hidden_size

    n_layers, hidden = extract_all(
        model, tokenizer, contexts, shard_dir, run_meta, force=args.force
    )

    print("[phase=b_bundle]", flush=True)
    gamma = {
        "input_layernorm": torch.stack(
            [layer.input_layernorm.weight.detach().float().cpu() for layer in model.model.layers]
        ),
        "post_attention_layernorm": torch.stack(
            [
                layer.post_attention_layernorm.weight.detach().float().cpu()
                for layer in model.model.layers
            ]
        ),
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
    }
    torch.save(gamma, out_dir / "rmsnorm_gamma.pt")

    raw_centroids: dict[str, torch.Tensor] = {}
    attn_centroids: dict[str, torch.Tensor] = {}
    mlp_centroids: dict[str, torch.Tensor] = {}
    per_probe = {"attn": {}, "mlp": {}, "raw": {}}
    dispersion: dict[str, dict] = {}
    for name in contexts:
        shard = torch.load(shard_dir / f"{name}.pt", weights_only=True)
        smeta = shard.get("meta") or {}
        assert smeta.get("probes_sha256") == probes_sha and smeta.get("model") == args.model, (
            name,
            "stale shard escaped the pre-pass — refusing to bundle",
            {k: smeta.get(k) for k in ("model", "n_probes", "probes_sha256", "dtype")},
        )
        raw_centroids[name] = shard["centroid_raw"]
        attn_centroids[name] = shard["centroid_attn"]
        mlp_centroids[name] = shard["centroid_mlp"]
        for kind in ("attn", "mlp", "raw"):
            per_probe[kind][name] = shard["per_probe_fp16"][kind]
        dispersion[name] = {
            kind: _dispersion(shard["per_probe_fp16"][kind].numpy())
            for kind in ("attn", "mlp", "raw")
        }

    meta = result_metadata(
        PROJECT_ROOT,
        extra={
            "phase": "B",
            "model": args.model,
            "n_layers": n_layers,
            "hidden": hidden,
            "n_contexts": len(contexts),
            "n_probes": len(probes),
            "probes_sha256": probes_sha,
            "device": device,
            "dtype": args.dtype,
        },
    )
    torch.save({"contexts": raw_centroids, "meta": meta}, out_dir / "context_vectors_all_layers.pt")
    torch.save(
        {"attn": attn_centroids, "mlp": mlp_centroids, "meta": meta},
        out_dir / "module_input_centroids.pt",
    )
    torch.save({**per_probe, "meta": meta}, out_dir / "per_probe_vectors_fp16.pt")
    (out_dir / "dispersion_diagnostics.json").write_text(
        json.dumps({"meta": meta, "contexts": dispersion}, indent=1)
    )
    manifest = {
        "meta": meta,
        "probe_set": "q_test_extended_50",
        "probes_sha256": probes_sha,
        "contexts": {
            name: {
                "groups": ctx["groups"],
                "system_prompt": ctx["system_prompt"],
                "prompt_sha256": ctx["prompt_sha256"],
                "first_prompt": ctx["prompts"][0],
            }
            for name, ctx in contexts.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    logger.info("bundle written to %s (%.1fs elapsed)", out_dir, time.time() - t0)

    if args.upload:
        print("[phase=b_upload]", flush=True)
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=f"{HF_BUCKET}/analysis_tensors",
            repo_id=args.upload_repo,
            repo_type="dataset",
            ignore_patterns=["shards/*"],
            commit_message="task #604 Phase B context-vector bundle",
        )
        listed = set(api.list_repo_files(args.upload_repo, repo_type="dataset"))
        required = [
            f"{HF_BUCKET}/analysis_tensors/{n}"
            for n in (
                "context_vectors_all_layers.pt",
                "module_input_centroids.pt",
                "per_probe_vectors_fp16.pt",
                "rmsnorm_gamma.pt",
                "dispersion_diagnostics.json",
                "manifest.json",
            )
        ]
        missing = [f for f in required if f not in listed]
        if missing:
            raise RuntimeError(f"upload verification FAILED — missing on Hub: {missing}")
        logger.info("upload verified: %s", info)

    # poll_pipeline sentinel (pod-side contract) — BEFORE [phase=done].
    if os.path.isdir("/workspace"):
        sentinel = {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 604,
            "by": "issue604_extract_context_vectors",
            "ts": meta["timestamp_utc"],
            "note": (
                "Phase B context-vector bundle complete: "
                f"{len(contexts)} contexts * {len(probes)} probes, {n_layers} layers; "
                f"uploaded={bool(args.upload)} repo={args.upload_repo} "
                f"path={HF_BUCKET}/analysis_tensors/"
            ),
        }
        log_dir = Path("/workspace/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"issue-604-epm_results-{int(time.time())}.json").write_text(
            json.dumps(sentinel, indent=1)
        )
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
