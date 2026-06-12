"""Issue #621 Phase B — context-vector bank at 3 positions × 5 taps (plan §4.3).

Extends the #604 extractor architecture (``scripts/issue604_extract_context_
vectors.py``, on main: input_layernorm/post_attention_layernorm forward
hooks + decoder-layer pre-forward hook, per-context shards with stale-shard
validation, fail-loud upload) into the issue-621 bank builder:

- **Contexts:** the 19-persona eval panel + the 4 unified-panel negatives,
  SHA-deduplicated by system prompt (assistant + kindergarten_teacher are
  panel members, so ≈21 unique contexts).
- **Generation pass (``--step generate``, vLLM):** greedy response per
  (context × probe), 512-new-token cap, truncation rate logged (>10% ⇒
  registered exclusion-sensitivity re-read per plan §8).
- **Capture pass (``--step capture``, HF):** ONE forward per
  (context, probe) over prompt + generated response, capturing at THREE
  positions — end_of_prompt (last prompt token), response_mean (mean over
  response content tokens), end_of_response (final content token; the
  prompt+raw-response construction carries no trailing specials) — for
  FIVE taps:
    raw      block input (pre-LN residual)            3584-d
    attn     input_layernorm output (q/v read this)   3584-d
    mlp      post_attention_layernorm output           3584-d
    o_in     o_proj module input (head-concat)         3584-d  ← NEW (write arm)
    down_in  down_proj module input (post-act hidden)  18944-d ← NEW (write arm)
  o_in/down_in use ``register_forward_pre_hook`` directly on the modules —
  no exemplar existed in the #604 file. Per-probe fp16 sidecars only for
  3584-d taps; the 18944-d space is centroids-only (plan §4.3).
- **Manifest assert (§14 duty 3):** the write-arm module-input spaces
  (o_in, down_in) MUST be present in the manifest's tap list + centroid
  bundle; the bundle build fails loud otherwise.

The two steps run as SEPARATE subprocesses (vLLM worker-orphan gotcha).
Smoke = the same entrypoints with ``--context-names`` + ``--probes 2`` +
a SEPARATE ``--out-dir``; the capture step also accepts
``--model Qwen/Qwen2.5-0.5B-Instruct --dtype float32`` for the CPU smoke
(dims are read from the model config; the 28/3584/18944 asserts apply only
to the production model).

CLI:
    uv run python scripts/issue621_extract_context_bank.py --step generate
    uv run python scripts/issue621_extract_context_bank.py --step capture --upload
"""

# ruff: noqa: RUF001, RUF002  # math notation in docstrings/labels

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_621 import (
    BANK_CAPTURE_POSITIONS,
    BANK_MAX_NEW_TOKENS,
    BANK_N_PROBES,
    BANK_SIDECAR_TAPS,
    BANK_TAPS,
    BANK_TRUNCATION_WARN_FRAC,
    BASE_MODEL,
    D_FF,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    HIDDEN_SIZE,
    N_LAYERS,
    PERSONA_POOL_19,
    UNIFIED_NEGATIVE_PANEL,
)
from explore_persona_space.experiments.issue_621.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)

log = logging.getLogger("issue_621.bank")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def assemble_contexts() -> dict[str, dict]:
    """Build the deduped context union: 19-persona eval panel + unified panel.

    Returns ``{name: {"groups": [...], "system_prompt": str,
    "prompt_sha256": str}}``. Byte-identical system prompts dedup (group
    membership recorded); DIFFERENT prompts under the same name fail loud.
    """
    bank = load_persona_bank()
    assert_registry_resolves(bank)

    contexts: dict[str, dict] = {}

    def _add(name: str, group: str) -> None:
        sp = bank[name]
        if name in contexts:
            if contexts[name]["system_prompt"] != sp:
                raise AssertionError(f"system-prompt drift for {name!r} across groups")
            contexts[name]["groups"].append(group)
            return
        # Dedup by prompt content across names too (byte-identical prompts
        # under two names would double-count one direction in the nulls).
        for other, ctx in contexts.items():
            if ctx["system_prompt"] == sp:
                raise AssertionError(
                    f"byte-identical system prompts for {name!r} and {other!r} — "
                    "the dedup contract (fixes #604's duplicate-bank defect) "
                    "refuses duplicate contexts under distinct names."
                )
        contexts[name] = {
            "groups": [group],
            "system_prompt": sp,
            "prompt_sha256": _sha256_text(sp),
        }

    for name in [*PERSONA_POOL_19, "assistant"]:
        _add(name, "eval_panel_19")
    for name in UNIFIED_NEGATIVE_PANEL:
        _add(name, "unified_negative_panel")
    return contexts


def _load_probes(n: int) -> list[str]:
    """The 50-probe set (q_test_extended_50, matching #604's bank)."""
    from explore_persona_space.experiments.i460_data import load_q_test_extended_50

    probes = load_q_test_extended_50()[:n]
    assert len(probes) == n, (len(probes), n)
    return probes


def _chat_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step: generate (vLLM)
# ─────────────────────────────────────────────────────────────────────────────


def step_generate(args) -> int:
    """vLLM greedy responses per (context × probe) → responses.json."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probes = _load_probes(args.probes)
    contexts = _select_contexts(args)

    prompts: list[str] = []
    meta: list[tuple[str, int]] = []
    for name, ctx in contexts.items():
        for pi, q in enumerate(probes):
            prompts.append(_chat_prompt(tokenizer, ctx["system_prompt"], q))
            meta.append((name, pi))
    log.info(
        "generate: %d contexts × %d probes = %d prompts", len(contexts), len(probes), len(prompts)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = {
            "dry_run": True,
            "n_prompts": len(prompts),
            "contexts": sorted(contexts),
            "first_prompt": prompts[0],
            "probes_sha256": _sha256_text("\x00".join(probes)),
        }
        (out_dir / "generate_dry_run.json").write_text(json.dumps(manifest, indent=1))
        log.info("generate --dry-run: wrote %s (no vLLM init)", out_dir / "generate_dry_run.json")
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        # prompts ~150 tokens + 512 new ⇒ 2048 holds with margin.
        max_model_len=2048,
        trust_remote_code=True,
    )
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens, seed=0)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    per_context: dict[str, dict] = {
        name: {"responses": [None] * len(probes), "truncated": [None] * len(probes)}
        for name in contexts
    }
    n_trunc = 0
    for (name, pi), output in zip(meta, outputs, strict=True):
        sample = output.outputs[0]
        text = sample.text
        truncated = len(sample.token_ids) >= args.max_new_tokens
        n_trunc += int(truncated)
        per_context[name]["responses"][pi] = text
        per_context[name]["truncated"][pi] = truncated

    trunc_rate = n_trunc / max(1, len(prompts))
    if trunc_rate > BANK_TRUNCATION_WARN_FRAC:
        log.warning(
            "TRUNCATION RATE %.1f%% > %.0f%% — plan §8 registered exclusion-"
            "sensitivity re-read applies; §13 authorizes ONE cap bump to 768 "
            "(re-run --step generate --max-new-tokens 768).",
            100 * trunc_rate,
            100 * BANK_TRUNCATION_WARN_FRAC,
        )

    payload = {
        "schema_version": "issue_621_bank_responses_v1",
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "n_probes": len(probes),
        "probes": probes,
        "probes_sha256": _sha256_text("\x00".join(probes)),
        "truncation_rate": trunc_rate,
        "contexts": {
            name: {
                "system_prompt": contexts[name]["system_prompt"],
                "groups": contexts[name]["groups"],
                "prompt_sha256": contexts[name]["prompt_sha256"],
                **per_context[name],
            }
            for name in contexts
        },
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "responses.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    log.info("generate: wrote %s (truncation %.1f%%)", out_dir / "responses.json", 100 * trunc_rate)
    return 0


def _select_contexts(args) -> dict[str, dict]:
    contexts = assemble_contexts()
    if args.context_names:
        wanted = [n.strip() for n in args.context_names.split(",") if n.strip()]
        missing = [n for n in wanted if n not in contexts]
        assert not missing, f"--context-names not in the assembled union: {missing}"
        contexts = {n: contexts[n] for n in wanted}
    return contexts


# ─────────────────────────────────────────────────────────────────────────────
# Step: capture (HF forward hooks)
# ─────────────────────────────────────────────────────────────────────────────


def _register_taps(model, captured: dict):
    """Register the 5 capture taps on every decoder layer. Returns handles.

    raw     — decoder-layer pre-forward hook (block input residual).
    attn    — forward hook on input_layernorm (output = what q/k/v read).
    mlp     — forward hook on post_attention_layernorm (what gate/up read).
    o_in    — forward PRE-hook on o_proj (input = head-concat, 3584-d).
    down_in — forward PRE-hook on down_proj (input = post-act MLP hidden,
              18944-d). No exemplar for module-input pre-hooks existed in
              the #604 extractor — these two are the #621 additions.
    """
    layers = model.model.layers
    handles = []

    def _ln_hook(kind: str, li: int):
        def hook(module, inputs, output):
            captured[(kind, li)] = output.detach()

        return hook

    def _block_pre_hook(li: int):
        def hook(module, args_, kwargs_):
            hs = args_[0] if args_ else kwargs_["hidden_states"]
            captured[("raw", li)] = hs.detach()

        return hook

    def _module_input_pre_hook(kind: str, li: int):
        def hook(module, args_):
            # o_proj/down_proj are called positionally inside Qwen2
            # attention/MLP forward; fail loud if that ever changes.
            if not args_:
                raise RuntimeError(f"{kind} pre-hook at layer {li}: empty positional args")
            captured[(kind, li)] = args_[0].detach()

        return hook

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(_block_pre_hook(li), with_kwargs=True))
        handles.append(layer.input_layernorm.register_forward_hook(_ln_hook("attn", li)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_ln_hook("mlp", li)))
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(_module_input_pre_hook("o_in", li))
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(_module_input_pre_hook("down_in", li))
        )
    return handles


def _validate_existing_shards(shard_dir: Path, contexts: dict, run_meta: dict) -> None:
    """Stale-shard validation pre-pass (#604 pattern) — BEFORE model load."""
    import torch

    stale: list[str] = []
    for name, ctx in contexts.items():
        sp = shard_dir / f"{name}.pt"
        if not sp.exists():
            continue
        smeta = torch.load(sp, weights_only=True).get("meta") or {}
        expected = {**run_meta, "prompt_sha256": ctx["prompt_sha256"]}
        bad = [
            f"{k}: shard={smeta.get(k)!r} != run={v!r}"
            for k, v in expected.items()
            if smeta.get(k) != v
        ]
        if bad:
            stale.append(f"{sp}: " + "; ".join(bad))
    if stale:
        raise RuntimeError(
            "stale bank shard(s) would be silently reused — refusing to resume:\n  "
            + "\n  ".join(stale)
            + "\nFix: use a SEPARATE --out-dir for smoke runs, pass --force, or "
            "delete the stale shards."
        )


def step_capture(args) -> int:  # noqa: C901  # hook + position bookkeeping
    """HF teacher-forced capture pass over prompt + generated response."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    responses_path = Path(args.responses_json or (out_dir / "responses.json"))
    if not responses_path.is_file():
        raise SystemExit(f"responses JSON missing at {responses_path}; run --step generate first.")
    gen = json.loads(responses_path.read_text())
    probes = gen["probes"]
    contexts_meta = gen["contexts"]
    if args.context_names:
        wanted = [n.strip() for n in args.context_names.split(",") if n.strip()]
        missing = [n for n in wanted if n not in contexts_meta]
        assert not missing, f"--context-names not in responses.json: {missing}"
        contexts_meta = {n: contexts_meta[n] for n in wanted}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    d_ff = model.config.intermediate_size
    if args.model == BASE_MODEL:
        assert n_layers == N_LAYERS, n_layers
        assert hidden == HIDDEN_SIZE, hidden
        assert d_ff == D_FF, d_ff
    tap_dims = {"raw": hidden, "attn": hidden, "mlp": hidden, "o_in": hidden, "down_in": d_ff}

    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    probes_sha = gen["probes_sha256"]
    run_meta = {
        "model": args.model,
        "n_probes": len(probes),
        "probes_sha256": probes_sha,
        "dtype": args.dtype,
        "responses_git_commit": gen.get("git_commit"),
    }
    if not args.force:
        _validate_existing_shards(shard_dir, contexts_meta, run_meta)

    captured: dict = {}
    handles = _register_taps(model, captured)
    print("[phase=bank_capture]", flush=True)
    t0 = time.time()
    try:
        for ci, (name, ctx) in enumerate(contexts_meta.items()):
            shard_path = shard_dir / f"{name}.pt"
            if shard_path.exists() and not args.force:
                log.info("[%d/%d] skip (shard exists): %s", ci + 1, len(contexts_meta), name)
                continue
            n_p = len(probes)
            per_probe = {
                (tap, pos): np.zeros((n_p, n_layers, tap_dims[tap]), dtype=np.float16)
                for tap in BANK_SIDECAR_TAPS
                for pos in BANK_CAPTURE_POSITIONS
            }
            sums = {
                (tap, pos): np.zeros((n_layers, tap_dims[tap]), dtype=np.float64)
                for tap in BANK_TAPS
                for pos in BANK_CAPTURE_POSITIONS
            }
            for pi, q in enumerate(probes):
                resp = ctx["responses"][pi]
                if resp is None:
                    raise AssertionError(f"context {name!r} probe {pi} has no response")
                prompt_text = _chat_prompt(tokenizer, ctx["system_prompt"], q)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                resp_ids = tokenizer.encode(resp, add_special_tokens=False)
                if len(resp_ids) < 1:
                    raise AssertionError(
                        f"context {name!r} probe {pi}: response tokenized to 0 ids "
                        f"(text head: {resp[:60]!r})"
                    )
                full_ids = prompt_ids + resp_ids
                P = len(prompt_ids)
                ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
                with torch.no_grad():
                    model(ids)
                # Position indices into the captured (1, T, d) tensors:
                # end_of_prompt = last prompt token; response span =
                # [P, T-1]; end_of_response = final content token (the
                # prompt+raw-response row has no trailing specials).
                pos_index = {
                    "end_of_prompt": P - 1,
                    "end_of_response": len(full_ids) - 1,
                }
                for tap in BANK_TAPS:
                    for li in range(n_layers):
                        t = captured[(tap, li)][0]  # (T, d)
                        assert t.shape == (len(full_ids), tap_dims[tap]), (
                            tap,
                            li,
                            tuple(t.shape),
                        )
                        for pos in BANK_CAPTURE_POSITIONS:
                            if pos == "response_mean":
                                vec = t[P:, :].float().mean(dim=0).cpu().numpy()
                            else:
                                vec = t[pos_index[pos], :].float().cpu().numpy()
                            assert np.isfinite(vec).all(), (name, tap, li, pos, "non-finite")
                            sums[(tap, pos)][li, :] += vec.astype(np.float64)
                            if tap in BANK_SIDECAR_TAPS:
                                v16 = vec.astype(np.float16)
                                assert np.isfinite(v16).all(), (
                                    name,
                                    tap,
                                    li,
                                    pos,
                                    "fp16 overflow in sidecar",
                                )
                                per_probe[(tap, pos)][pi, li, :] = v16
                captured.clear()
            import torch as _torch

            shard = {
                "name": name,
                "groups": ctx["groups"],
                "meta": {
                    **run_meta,
                    "n_layers": n_layers,
                    "hidden": hidden,
                    "d_ff": d_ff,
                    "prompt_sha256": ctx["prompt_sha256"],
                },
                "centroids": {
                    f"{tap}|{pos}": _torch.from_numpy((sums[(tap, pos)] / n_p).astype(np.float32))
                    for tap in BANK_TAPS
                    for pos in BANK_CAPTURE_POSITIONS
                },
                "per_probe_fp16": {
                    f"{tap}|{pos}": _torch.from_numpy(per_probe[(tap, pos)])
                    for tap in BANK_SIDECAR_TAPS
                    for pos in BANK_CAPTURE_POSITIONS
                },
            }
            _torch.save(shard, shard_path)
            log.info("[%d/%d] context %s done", ci + 1, len(contexts_meta), name)
    finally:
        for h in handles:
            h.remove()

    print("[phase=bank_bundle]", flush=True)
    import torch as _torch

    gamma = {
        "input_layernorm": _torch.stack(
            [layer.input_layernorm.weight.detach().float().cpu() for layer in model.model.layers]
        ),
        "post_attention_layernorm": _torch.stack(
            [
                layer.post_attention_layernorm.weight.detach().float().cpu()
                for layer in model.model.layers
            ]
        ),
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
    }
    _torch.save(gamma, out_dir / "rmsnorm_gamma.pt")

    centroids: dict[str, dict[str, dict[str, object]]] = {
        tap: {pos: {} for pos in BANK_CAPTURE_POSITIONS} for tap in BANK_TAPS
    }
    per_probe_bundle: dict[str, dict[str, dict[str, object]]] = {
        tap: {pos: {} for pos in BANK_CAPTURE_POSITIONS} for tap in BANK_SIDECAR_TAPS
    }
    dispersion: dict[str, dict] = {}
    for name in contexts_meta:
        shard = _torch.load(shard_dir / f"{name}.pt", weights_only=True)
        smeta = shard.get("meta") or {}
        assert smeta.get("probes_sha256") == probes_sha and smeta.get("model") == args.model, (
            name,
            "stale shard escaped the pre-pass — refusing to bundle",
        )
        for tap in BANK_TAPS:
            for pos in BANK_CAPTURE_POSITIONS:
                centroids[tap][pos][name] = shard["centroids"][f"{tap}|{pos}"]
        for tap in BANK_SIDECAR_TAPS:
            for pos in BANK_CAPTURE_POSITIONS:
                per_probe_bundle[tap][pos][name] = shard["per_probe_fp16"][f"{tap}|{pos}"]
        dispersion[name] = {
            f"{tap}|{pos}": _split_half_cos(shard["per_probe_fp16"][f"{tap}|{pos}"].numpy())
            for tap in BANK_SIDECAR_TAPS
            for pos in BANK_CAPTURE_POSITIONS
        }

    meta = {
        "phase": "bank",
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "d_ff": d_ff,
        "n_contexts": len(contexts_meta),
        "n_probes": len(probes),
        "probes_sha256": probes_sha,
        "positions": list(BANK_CAPTURE_POSITIONS),
        "taps": list(BANK_TAPS),
        "sidecar_taps": list(BANK_SIDECAR_TAPS),
        "truncation_rate": gen.get("truncation_rate"),
        "dtype": args.dtype,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    _torch.save({"centroids": centroids, "meta": meta}, out_dir / "centroids.pt")
    _torch.save({"per_probe": per_probe_bundle, "meta": meta}, out_dir / "per_probe_fp16.pt")
    (out_dir / "dispersion_diagnostics.json").write_text(
        json.dumps({"meta": meta, "contexts": dispersion}, indent=1)
    )
    manifest = {
        "meta": meta,
        "probe_set": "q_test_extended_50",
        "contexts": {
            name: {
                "groups": ctx["groups"],
                "system_prompt": ctx["system_prompt"],
                "prompt_sha256": ctx["prompt_sha256"],
                "truncated_count": sum(bool(t) for t in ctx["truncated"]),
            }
            for name, ctx in contexts_meta.items()
        },
    }
    # §14 duty 3: the write-arm module-input spaces MUST be in the manifest.
    for required_tap in ("o_in", "down_in"):
        if required_tap not in manifest["meta"]["taps"]:
            raise AssertionError(
                f"manifest tap list {manifest['meta']['taps']} is missing the "
                f"write-arm space {required_tap!r} (§14 duty 3)."
            )
        if not centroids[required_tap]["end_of_response"]:
            raise AssertionError(
                f"centroid bundle has no contexts for write-arm tap {required_tap!r}."
            )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("bundle written to %s (%.1fs elapsed)", out_dir, time.time() - t0)

    if args.upload:
        print("[phase=bank_upload]", flush=True)
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=HF_ANALYSIS_TENSORS_PREFIX,
            repo_id=args.upload_repo,
            repo_type="dataset",
            ignore_patterns=["shards/*"],
            commit_message="task #621 context-vector bank (3 positions × 5 taps)",
        )
        listed = set(api.list_repo_files(args.upload_repo, repo_type="dataset"))
        required = [
            f"{HF_ANALYSIS_TENSORS_PREFIX}/{n}"
            for n in (
                "centroids.pt",
                "per_probe_fp16.pt",
                "rmsnorm_gamma.pt",
                "dispersion_diagnostics.json",
                "manifest.json",
                "responses.json",
            )
        ]
        missing = [f for f in required if f not in listed]
        if missing:
            raise RuntimeError(f"bank upload verification FAILED — missing on Hub: {missing}")
        log.info("bank upload verified (%d required files).", len(required))
    return 0


def _split_half_cos(per_probe: np.ndarray) -> list[float]:
    """Split-half centroid cosine per layer (dispersion diagnostic)."""
    p, n_layers, _d = per_probe.shape
    arr = per_probe.astype(np.float32)
    half = p // 2
    out: list[float] = []
    for li in range(n_layers):
        x = arr[:, li, :]
        a, b = x[:half].mean(0), x[half:].mean(0)
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
        out.append(float(a @ b / denom))
    return out


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", required=True, choices=["generate", "capture"])
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--probes", type=int, default=BANK_N_PROBES)
    ap.add_argument(
        "--context-names",
        default="",
        help="comma-separated explicit context subset (smoke; same loop as the full set)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=BANK_MAX_NEW_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="float32 for the CPU smoke; production stays bf16",
    )
    ap.add_argument(
        "--responses-json",
        default=None,
        help="capture only — override the responses.json path (CPU smoke fixtures)",
    )
    ap.add_argument("--force", action="store_true", help="recompute existing shards in place")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="generate only — build prompts + manifest WITHOUT initializing vLLM",
    )
    ap.add_argument("--upload", action="store_true", help="capture only — push bundle to HF")
    ap.add_argument("--upload-repo", default=HF_DATA_REPO)
    ap.add_argument("--out-dir", default="eval_results/issue_621/context_vectors")
    args = ap.parse_args(argv)
    assert args.probes >= 2, "need >= 2 probes (split-half dispersion diagnostics)"

    if args.step == "generate":
        return step_generate(args)
    return step_capture(args)


if __name__ == "__main__":
    sys.exit(main())
