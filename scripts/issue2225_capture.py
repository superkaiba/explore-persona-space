#!/usr/bin/env python
"""Issue #2225 Phase 2d — teacher-forced activation capture over the P2b rollouts.

Plan §4.6 item 4: for EVERY eval rollout from P2b (``issue2225_eval_gen``
outputs), per-response activation summaries at THREE positions — response-avg,
context-end (last prompt token), prefix-end (last SYSTEM-segment token) — at
all 28 layers, stored fp16. The capture helpers are IMPORTED, never copied:

  - response-avg:  ``issue778_lib.capture_response_avg_all_layers``  (:289)
  - context-end:   ``issue778_lib.capture_last_prompt_token_all_layers`` (:334)
  - prefix-end:    ``experiments.issue2225.directions.capture_prefix_end_all_layers``
                   (unit 1's E3 sibling; issue1415 ``prefix_end_index`` on token
                   IDS — the eval prompt has no system turn, so the prefix is
                   Qwen's implicit default system segment, plan §12 A15)

Context-end + prefix-end depend only on the PROMPT, so they are captured once
per question (all rollouts under a context share them — the plan §4.2 E2
premise); response-avg is per rollout.

BPE-SEAM AUDIT (gotchas.md teacher-forced capture rig): the reused #778
response-avg helper tokenizes ``prompt + response`` as a concatenated STRING
with ``prompt_len`` from a separate encode (paper-faithful
``get_hidden_p_and_r``). Rather than fork the plan-named helper, this script
counts the rows where ``enc(prompt + response)[:len(enc(prompt))] !=
enc(prompt)`` (a seam merge shifting the response boundary) and records
``seam_mismatch_count`` per (target, trait) in the summary manifest — a
nonzero count is a disclosed fidelity caveat for the analyst, never silent.

STATED DEVIATION from plan §4.6 item 4 (g3 Major 2, carried into the
clean-result scope caveats): the plan's wording claims "per-segment token-id
concatenation", but the plan-NAMED helper is the string-concat form above.
The reuse is DELIBERATE — #778's probe-pool activations (the P5 probe's
training data) were captured with the SAME helper, so forking to ids-concat
here would create a probe-train / probe-apply capture-convention mismatch
(agent-memory: capture convention = read the PRODUCER's code). The P5
analysis CONSUMES ``seam_mismatch_count`` (``issue2225_analysis.py``
run_probe seam_audit block): seam-flagged units are enumerated + fractioned
in ``probe_shifts.json`` for exclusion / sensitivity reads.

OUTPUTS per model: ``<out-root>/capture/<tag>/<trait>.pt`` (fp16 tensors +
row-alignment indices) + ``<trait>.meta.json`` (the resume fingerprint sidecar)
+ ``summary_manifest.json`` (the plan §6.5 deliverable). ``--upload`` pushes
``capture/`` to ``issue2225_ctxsteer/analysis_tensors/capture/`` — one folder
commit by default, or per-model commits via ``--tags`` (the §9 per-chunk
upload).

RESUME (#952 shape): a (target, trait) is skipped iff its ``.pt`` exists AND
the ``.meta.json`` fingerprint (adapter sha + P2b input sha + positions +
dtype) matches. GPU-sharded per model via the shared CVD-pinned work-stealing
fan-out (one HF model load per ``--single`` subprocess).

CONTENT HYGIENE: rollout text is harmful content — never printed; progress
lines carry counts only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# scripts/ on sys.path so the sibling issue778_* / issue2225_* modules resolve
# in script mode (the #823 sys.path[0] trap). Heavy imports stay deferred.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2225.capture")
load_dotenv()

import issue2225_eval_gen as evalgen  # targets + adapter resolution + fan-out
import issue2225_train as train  # sha helper (cheap import)
import issue778_lib as lib  # capture helpers + constants (cheap import)

# ── constants ────────────────────────────────────────────────────────────────

DATA_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/capture"
POSITIONS = ("response_avg", "context_end", "prefix_end")
STORE_DTYPE = "float16"  # plan §4.6 item 4


# ── fingerprint + resume ─────────────────────────────────────────────────────


def capture_fingerprint(
    target: evalgen.EvalTarget,
    trait: str,
    adapter_path: Path | None,
    gen_path: Path,
    *,
    model: str,
) -> dict:
    """Resume-compared fingerprint: adapter sha + P2b input sha + capture recipe
    + base model (every output-affecting regime key, #722 r3 / g3 minor)."""
    if adapter_path is None:
        adapter_sha = "base-no-adapter"
    else:
        adapter_sha = train._sha256(adapter_path / "adapter_model.safetensors")
    return {
        "tag": target.tag,
        "trait": trait,
        "adapter_sha256": adapter_sha,
        "gen_input_sha256": train._sha256(gen_path),
        "model": model,
        "positions": list(POSITIONS),
        "n_layers": lib.N_LAYERS,
        "dtype": STORE_DTYPE,
    }


def trait_pt_path(out_root: Path, tag: str, trait: str) -> Path:
    return out_root / "capture" / tag / f"{trait}.pt"


def _trait_done(pt_path: Path, fingerprint: dict) -> bool:
    meta_path = pt_path.with_suffix(".meta.json")
    if not (pt_path.exists() and meta_path.exists()):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("[resume] unreadable meta %s (%s) -> re-run", meta_path, e)
        return False
    return stored.get("fingerprint") == fingerprint


# ── seam audit (report-only; the reused helper stays paper-faithful) ─────────


def seam_audit(tokenizer, prompts: Sequence[str], responses: Sequence[str]) -> int:
    """Count rows where the prompt+response concatenation BPE-merges across the
    seam (``enc(prompt+response)`` no longer starts with ``enc(prompt)``) —
    those rows' response-avg boundary is shifted by the reused #778 helper."""
    mismatches = 0
    for prompt, response in zip(prompts, responses, strict=True):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + response, add_special_tokens=False)
        if full_ids[: len(p_ids)] != p_ids:
            mismatches += 1
    return mismatches


# ── single-model capture (subprocess mode) ───────────────────────────────────


def capture_one_model(args) -> None:
    """Capture all pending (trait) units for ONE target on the pinned GPU."""
    # Registry lookup with the resolve_targets cell fallback (fu1 seam):
    # fu1 cells capture without a registry edit; unknown tags still fail loud.
    target = evalgen.resolve_targets([args.single])[0]
    out_root = Path(args.out_root)
    gen_root = Path(args.gen_root) if args.gen_root else out_root
    model_name = args.model or lib.MODEL_NAME
    adapter = evalgen.resolve_adapter(
        target, ckpt_root=Path(args.ckpt_root), staging_dir=Path(args.staging_dir)
    )

    # Enumerate pending traits (fail loud on a missing P2b input — the capture
    # consumes P2b's REAL output, never a substitute).
    units: list[tuple[str, Path, dict]] = []
    n_skipped = 0
    for trait in target.traits:
        gen_path = evalgen.unit_out_path(gen_root, target, trait)
        if not gen_path.exists():
            raise FileNotFoundError(
                f"P2b output missing for {target.tag}__{trait}: {gen_path} "
                f"(run issue2225_eval_gen.py first)"
            )
        fp = capture_fingerprint(target, trait, adapter, gen_path, model=model_name)
        pt_path = trait_pt_path(out_root, target.tag, trait)
        if _trait_done(pt_path, fp):
            n_skipped += 1
            print(f"[capture] skip {target.tag}__{trait} (resume)", flush=True)
            continue
        units.append((trait, gen_path, fp))

    if not units:
        print(f"[capture] {target.tag}: nothing pending ({n_skipped} skipped)", flush=True)
        _write_model_manifest(out_root, target, adapter)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue2225.directions import (
        capture_prefix_end_all_layers,
    )

    lib.log_phase("capture", f"model={target.tag} start ({len(units)} traits)", model=target.tag)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()  # merged for a clean forward (#778 capture pattern)

    try:
        total = len(units)
        for k, (trait, gen_path, fp) in enumerate(units, start=1):
            t0 = time.time()
            with open(gen_path, encoding="utf-8") as f:
                payload = json.load(f)
            questions = [row["question"] for row in payload["rows"]]
            n_rollouts = payload["n_rollouts"]
            prompts_by_q = [evalgen._chat_prompt(tokenizer, q) for q in questions]

            flat_prompts: list[str] = []
            flat_responses: list[str] = []
            q_idx: list[int] = []
            r_idx: list[int] = []
            for qi, row in enumerate(payload["rows"]):
                if len(row["rollouts"]) != n_rollouts:
                    raise ValueError(
                        f"{target.tag}__{trait} q{qi}: {len(row['rollouts'])} rollouts, "
                        f"expected {n_rollouts}"
                    )
                for ri, response in enumerate(row["rollouts"]):
                    flat_prompts.append(prompts_by_q[qi])
                    flat_responses.append(response)
                    q_idx.append(qi)
                    r_idx.append(ri)

            n_seam = seam_audit(tokenizer, flat_prompts, flat_responses)
            if n_seam:
                logger.warning(
                    "[seam-audit] %s__%s: %d/%d rows BPE-merge at the prompt/response "
                    "seam (response-avg boundary shifted on those rows — disclosed)",
                    target.tag,
                    trait,
                    n_seam,
                    len(flat_prompts),
                )

            # Per-rollout response-avg (reused #778 helper, paper-faithful).
            resp_avg = lib.capture_response_avg_all_layers(
                model, tokenizer, flat_prompts, flat_responses, device=model.device
            )
            # Per-question context-end + prefix-end (shared across rollouts).
            ctx_end = lib.capture_last_prompt_token_all_layers(
                model, tokenizer, prompts_by_q, device=model.device
            )
            pfx_end = capture_prefix_end_all_layers(
                model, tokenizer, prompts_by_q, device=model.device
            )

            store = {
                "response_avg": resp_avg.to(torch.float16),  # (n_q*n_r, 28, 3584)
                "context_end": ctx_end.to(torch.float16),  # (n_q, 28, 3584)
                "prefix_end": pfx_end.to(torch.float16),  # (n_q, 28, 3584)
                "question_idx": torch.tensor(q_idx, dtype=torch.long),
                "rollout_idx": torch.tensor(r_idx, dtype=torch.long),
                "fingerprint": fp,
            }
            pt_path = trait_pt_path(out_root, target.tag, trait)
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = pt_path.with_name(pt_path.stem + ".tmp.pt")
            torch.save(store, tmp)
            tmp.replace(pt_path)
            meta = {
                "model_tag": target.tag,
                "trait": trait,
                "n_rows": len(flat_prompts),
                "n_questions": len(questions),
                "n_rollouts": n_rollouts,
                "seam_mismatch_count": n_seam,
                "shapes": {k: list(v.shape) for k, v in store.items() if hasattr(v, "shape")},
                "fingerprint": fp,
                "reproducibility": lib.repro_metadata(),
            }
            evalgen._atomic_write_json(pt_path.with_suffix(".meta.json"), meta)
            print(
                f"[capture] unit {k}/{total} {target.tag}__{trait} "
                f"rows={len(flat_prompts)} seam_mismatch={n_seam} "
                f"elapsed={round(time.time() - t0, 1)}s",
                flush=True,
            )
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    _write_model_manifest(out_root, target, adapter)
    lib.log_phase("capture", f"model={target.tag} done", model=target.tag)


def _write_model_manifest(out_root: Path, target: evalgen.EvalTarget, adapter: Path | None) -> None:
    """The plan §6.5 ``summary_manifest.json`` per model (from the trait sidecars)."""
    model_dir = out_root / "capture" / target.tag
    traits: dict[str, dict] = {}
    for trait in target.traits:
        meta_path = trait_pt_path(out_root, target.tag, trait).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                traits[trait] = json.load(f)
    manifest = {
        "model_tag": target.tag,
        "kind": target.kind,
        "dataset": target.dataset,
        "adapter_path": (None if adapter is None else str(adapter)),
        "positions": list(POSITIONS),
        "dtype": STORE_DTYPE,
        "n_layers": lib.N_LAYERS,
        "hidden_dim": lib.HIDDEN_DIM,
        "traits_present": sorted(traits),
        "traits_expected": list(target.traits),
        "traits": traits,
        "reproducibility": lib.repro_metadata(),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    evalgen._atomic_write_json(model_dir / "summary_manifest.json", manifest)


# ── fan-out over targets ─────────────────────────────────────────────────────


def run_fan_out(args) -> None:
    if args.targets:
        wanted = [s.strip() for s in args.targets.split(",") if s.strip()]
        # Registry lookup with the resolve_cell fallback (fu1 seam) — fu1 cells
        # capture without a registry edit; unknown tags still fail loud.
        targets = evalgen.resolve_targets(wanted)
    else:
        targets = evalgen.build_eval_targets()
    if args.smoke:
        targets = [t for t in targets if t.kind == "base"] or targets[:1]

    if args.dry_run:
        n_gpus = args.n_gpus or 8
    else:
        n_gpus = train._detect_gpu_count(cpu_only=False)
        if args.n_gpus:
            n_gpus = max(1, min(n_gpus, args.n_gpus))

    out_root = Path(args.out_root)

    def build_cmd(tag: str, gpu_id: int) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--single",
            tag,
            "--gpu-id",
            str(gpu_id),
            "--out-root",
            str(out_root),
            "--ckpt-root",
            str(args.ckpt_root),
            "--staging-dir",
            str(args.staging_dir),
        ]
        if args.gen_root:
            cmd += ["--gen-root", str(args.gen_root)]
        if args.model:
            cmd += ["--model", args.model]
        return cmd

    if args.dry_run:
        for i, t in enumerate(targets):
            g = i % n_gpus
            print(f"[capture][dry-run] CUDA_VISIBLE_DEVICES={g} {' '.join(build_cmd(t.tag, g))}")
        return

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(out_root, 5.0, phase="p2d-capture")
    lib.log_phase("capture", f"fan-out {len(targets)} targets over {n_gpus} GPUs")
    evalgen._prestage_base_model(args.model or lib.MODEL_NAME)
    evalgen.fan_out_subprocesses(
        [t.tag for t in targets],
        build_cmd,
        n_gpus=n_gpus,
        log_dir=out_root / "logs" / "capture",
        label="capture",
    )
    lib.log_phase("capture", "fan-out complete")


# ── upload ───────────────────────────────────────────────────────────────────


# Parent-default-identical seam: parent #2225 call sites pass no prefix and must keep it.
# UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; fu1 threads its own via --hf-prefix
def upload_capture(
    out_root: Path,
    tags: Sequence[str] | None = None,
    *,
    hf_prefix: str = CAPTURE_HF_PREFIX,
    hf_repo: str = DATA_REPO,
) -> list[str]:
    """Upload capture tensors to the HF data repo. Default: ONE folder commit of
    ``capture/``; ``tags`` uploads per-model subdirs (the §9 per-chunk path).
    Follow-up rounds thread their OWN prefix (fu1: analysis_tensors/fu1_capture
    — never the parent-clobbering default, #1452). ``hf_repo`` (default: the
    canonical data repo) lets a round route the capture class to the private
    overflow repo under the SAME prefix layout when the canonical repo is at
    the 1M-file ceiling (#1108 contract; fu1 round 5)."""
    from explore_persona_space.orchestrate.hub import _upload

    urls: list[str] = []
    if tags:
        for tag in tags:
            local = out_root / "capture" / tag
            if not local.exists():
                raise FileNotFoundError(f"capture dir missing for upload: {local}")
            url = _upload(local, hf_repo, "dataset", f"{hf_prefix}/{tag}", raise_on_error=True)
            print(f"[capture] uploaded {local} -> {url}", flush=True)
            urls.append(url)
        return urls
    local = out_root / "capture"
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload: {local} absent")
    url = _upload(local, hf_repo, "dataset", hf_prefix, raise_on_error=True)
    print(f"[capture] uploaded {local} -> {url}", flush=True)
    return [url]


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 Phase 2d activation capture.")
    ap.add_argument("--out-root", default="data/issue_2225/p2b_out")
    ap.add_argument("--gen-root", default=None, help="P2b out-root (default: --out-root)")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225")
    ap.add_argument("--staging-dir", default="data/issue_2225/hf_dl/eval_adapters")
    ap.add_argument("--targets", default=None, help="comma-separated target tags (default: all)")
    ap.add_argument("--model", default=None, help="base model (default: issue778_lib.MODEL_NAME)")
    ap.add_argument("--single", default=None, help="capture ONE target by tag (subprocess mode)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width cap")
    ap.add_argument("--smoke", action="store_true", help="base target only")
    ap.add_argument("--dry-run", action="store_true", help="print invocations, no CUDA")
    ap.add_argument("--upload", action="store_true", help="upload-only mode (pod-side, later)")
    ap.add_argument("--upload-tags", default=None, help="with --upload: per-model subdirs")
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam — issue2225's own dispatcher calls
    # this flag-less and must keep the parent prefix; fu1 rounds pass an explicit --hf-prefix.
    ap.add_argument(
        "--hf-prefix",
        default=CAPTURE_HF_PREFIX,
        help="HF prefix for the capture upload (fu rounds thread analysis_tensors/fu1_capture)",
    )
    ap.add_argument(
        "--hf-repo",
        default=DATA_REPO,
        help=(
            "HF dataset repo for the capture upload (default: canonical data repo; "
            "fu1 threads the private overflow repo when the canonical repo is at "
            "the 1M-file ceiling — #1108 contract, same prefix layout)"
        ),
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the production paths reach (#606).
        import torch  # noqa: F401
        from peft import PeftModel  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.experiments.issue1415.steering import (  # noqa: F401
            prefix_end_index,
        )
        from explore_persona_space.experiments.issue2225.directions import (  # noqa: F401
            capture_prefix_end_all_layers,
        )
        from explore_persona_space.orchestrate.hub import _upload  # noqa: F401
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )

        evalgen.build_eval_targets()  # asserts 86 / 67 / 19 / 124
        print("[issue2225-capture] import-check OK", flush=True)
        raise SystemExit(0)

    if args.upload:
        # `if s.strip()`: a trailing comma must not yield tag "" — that would
        # resolve out_root/capture/"" to the WHOLE capture dir and malform the
        # dest prefix with a trailing slash (g3 minor).
        tags = (
            [s.strip() for s in args.upload_tags.split(",") if s.strip()]
            if args.upload_tags
            else None
        )
        # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam; fu1 passes an explicit --hf-prefix
        upload_capture(Path(args.out_root), tags, hf_prefix=args.hf_prefix, hf_repo=args.hf_repo)
        sys.stdout.flush()
        sys.exit(0)

    if args.single:
        capture_one_model(args)
    else:
        run_fan_out(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
