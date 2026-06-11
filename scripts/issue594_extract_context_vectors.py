#!/usr/bin/env python3
"""Issue #594 Phase 1: extract per-layer context vectors for the battery.

Per plan v1 §3 Phase 1. For every battery instance x every preregistered
probe, forward-pass the chat-templated prompt through Qwen-2.5-7B-Instruct
(bf16, batch=1, no padding) and capture the residual-stream activation at the
LAST input token (the newline of ``<|im_start|>assistant\\n`` under
``add_generation_prompt=True``) at ALL decoder layers via forward hooks on
``model.model.layers[i]`` (pre-final-norm; NOT ``output_hidden_states`` —
its final element is post-final-norm and would break layer-27 comparability
with the #404/#468 hook reads).

Outputs (``--out-dir``, checkpoint-per-instance with resume-skip):

- ``per_probe/<instance_id>.pt``  fp16 (P, L, H) per-probe tensors
- ``context_vectors_mean.pt``     fp32 (N, L, H) probe-mean tensor + index
- ``extraction_manifest.json``    per-instance provenance + length covariates

``--smoke`` runs the IDENTICAL code path end-to-end on 4 instances — one per
structural template shape (system-prompt persona, ICL prefix-messages,
WildChat multi-turn, bare default) — x 4 probes into ``<out-dir>_smoke``,
plus a tiny HF upload probe (plan §14 item 5: smoke IS the pipeline with
tiny N; no separate smoke architecture).

Follow-up ``probe-genre-generalization`` flags (plan v2 §4; flag-routing
ONLY — zero changes to the capture path):

- ``--probes-file``: load the probe pool from a builder JSON (e.g.
  ``data/issue594/probes_ultrachat.json``) instead of the Betley
  preregistered set; asserts the file's own ``meta.probe_pool_hash``,
  bypasses the battery-meta Betley assert with a logged notice, records
  ``probe_pool_source: probes_file`` (+ path + hash) in the manifest, and
  uploads the probes file to ``<prefix>/inputs/`` alongside the battery.
- ``--hf-subdir``: verbatim HF upload sub-directory (replaces the hardcoded
  ``analysis_tensors`` / ``smoke_probe`` choice); the v2 launch passes
  ``--hf-subdir analysis_tensors_probegen``. Smoke runs should pass their
  own smoke subdir (or omit the flag for the default ``smoke_probe``).

Pod-side contract: emits ``[phase=...]`` log lines ending in ``[phase=done]``
and writes a ``poll_pipeline.py``-conformant end-of-run sentinel.

Usage (plan §8 launch command)::

    nohup uv run python scripts/issue594_extract_context_vectors.py \\
        --battery data/issue594/battery.json \\
        --out-dir data/issue594/context_vectors --gpu-id 0 \\
        > logs/issue594_extract.log 2>&1 &

    # pod smoke: same + --smoke
    # local CPU smoke (tiny same-template-family model):
    uv run python scripts/issue594_extract_context_vectors.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --expected-layers 24 \\
        --expected-hidden 896 --device cpu \\
        --out-dir /tmp/issue594_cpu_smoke --no-upload --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    fetch_betley_main_8,
    fetch_preregistered_probes,
    reproducibility_metadata,
)
from issue594_common import (  # noqa: E402
    BATTERY_PATH,
    DEFAULT_MODEL,
    DEFAULT_VECTORS_DIR,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    HF_PREFIX,
    load_battery,
    messages_for_instance,
    probes_hash,
)

load_dotenv()

logger = logging.getLogger("issue594_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GENERATION_SUFFIX = "<|im_start|>assistant\n"
SENTINEL_SCHEMA_VERSION = 1


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line (PHASE_RE on the log tail)."""
    print(f"[phase={name}]", flush=True)


# ── Hook capture ─────────────────────────────────────────────────────────────


class LayerCapture:
    """Forward hooks on every decoder block; keeps the latest (1, T, H) per layer."""

    def __init__(self, model, n_layers: int):
        self.latest: dict[int, torch.Tensor] = {}
        self._handles = []
        for li in range(n_layers):
            self._handles.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.latest[layer_idx] = hs.detach()

        return hook_fn

    def last_token_stack(self, n_layers: int) -> torch.Tensor:
        """(L, H) fp32 CPU stack of the last-position activation per layer."""
        vecs = [self.latest[li][0, -1, :].float().cpu() for li in range(n_layers)]
        self.latest.clear()
        return torch.stack(vecs)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def extract_for_instance(
    model,
    tokenizer,
    instance: dict,
    probes: list[str],
    capture: LayerCapture,
    n_layers: int,
) -> tuple[torch.Tensor, list[int]]:
    """Forward every probe under the instance's context; capture all layers.

    Returns (per_probe fp32 (P, L, H), per-probe total prompt token counts).
    Per-forward position assert: the last 3 input tokens must decode to the
    assistant-header suffix — fail LOUD on any drift (plan §4 control; kill
    criterion §9(a) is implemented as fail-fast, stricter than the 10% bound,
    because a mixed-position tensor must never be analyzed).
    """
    per_probe: list[torch.Tensor] = []
    token_lens: list[int] = []
    for q in probes:
        messages = messages_for_instance(instance, q)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        suffix = tokenizer.decode(inputs["input_ids"][0, -3:])
        assert suffix == GENERATION_SUFFIX, (
            f"position assert failed for instance={instance['id']} probe={q[:40]!r}: "
            f"last-3-token decode {suffix!r} != {GENERATION_SUFFIX!r}"
        )
        with torch.no_grad():
            _ = model(**inputs)
        per_probe.append(capture.last_token_stack(n_layers))
        token_lens.append(int(inputs["input_ids"].shape[1]))
    stacked = torch.stack(per_probe)  # (P, L, H) fp32
    assert stacked.shape == (len(probes), n_layers, stacked.shape[-1]), stacked.shape
    return stacked, token_lens


def default_template_token_lens(tokenizer, instances: list[dict], probes: list[str]) -> list[int]:
    """Tokenize-only token counts for the bare default-template instance.

    Used for the plan-§3 delta covariate
    (median_p[len(instance, p) - len(default, p)]).
    """
    default = next(i for i in instances if i["id"] == "f6_default_template")
    lens = []
    for q in probes:
        text = tokenizer.apply_chat_template(
            messages_for_instance(default, q), tokenize=False, add_generation_prompt=True
        )
        lens.append(len(tokenizer(text, padding=False)["input_ids"]))
    return lens


def content_token_len(tokenizer, instance: dict) -> int:
    """Context-content token count: system prompt + prefix-message contents.

    >=0 always; exactly 0 for the bare default template. This is the length
    covariate the §14-item-2 log1p reads use — the plan-§3 delta-vs-default
    covariate is ALSO recorded, but it goes NEGATIVE for system prompts
    shorter than Qwen's injected default (e.g. the terse rephrase), which
    would NaN under log1p. Deviation recorded in the manifest + report.
    """
    n = 0
    if instance["system_prompt"] is not None:
        n += len(tokenizer.encode(instance["system_prompt"], add_special_tokens=False))
    for m in instance["prefix_messages"]:
        n += len(tokenizer.encode(m["content"], add_special_tokens=False))
    return n


# ── Manifest / sentinel ──────────────────────────────────────────────────────


def write_manifest(path: Path, manifest: dict) -> None:
    """Atomic-ish manifest rewrite (tmp + rename) after every instance."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


def write_sentinel(kind: str, note: str, task_id: int = 594) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue594_extract_context_vectors",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


# ── HF upload ────────────────────────────────────────────────────────────────


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def upload_outputs(
    out_dir: Path, smoke: bool, expected_per_probe: int, hf_subdir: str | None = None
) -> dict:
    """Bulk-upload the extraction outputs to the HF data repo and verify.

    ONE ``upload_folder`` commit (well under the 256/hr cap), verified via
    ``huggingface_hub.list_repo_files`` (never the ``hf`` CLI — false "0
    files"). On the account-wide LFS storage-quota 403, falls back to the
    private overflow repo per .claude/rules/upload-policy.md and records the
    deviation. Fail-loud otherwise. Verification asserts the REMOTE per-probe
    file count >= ``expected_per_probe`` (the §6.5 primary deliverables), not
    just the mean tensor + manifest.

    ``hf_subdir`` (plan v2 §4): when set, used VERBATIM as the upload
    sub-directory, replacing the hardcoded smoke/full choice.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    sub = hf_subdir or ("smoke_probe" if smoke else "analysis_tensors")
    path_in_repo = f"{HF_PREFIX}/{sub}"
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue594: {'smoke probe' if smoke else 'context vectors'} upload",
        )
    except Exception as e:
        if not _is_storage_quota_403(e):
            raise
        logger.warning("HF storage-quota 403 on %s; falling back to overflow repo", HF_DATA_REPO)
        repo_used = HF_OVERFLOW_REPO
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_OVERFLOW_REPO,
            repo_type="dataset",
            commit_message="issue594: context vectors upload (quota-403 overflow fallback)",
        )
    files = [
        f for f in api.list_repo_files(repo_used, repo_type="dataset") if f.startswith(path_in_repo)
    ]
    expected = {
        f"{path_in_repo}/context_vectors_mean.pt",
        f"{path_in_repo}/extraction_manifest.json",
    }
    missing = expected - set(files)
    if missing:
        raise RuntimeError(f"upload verification failed; missing on {repo_used}: {missing}")
    n_per_probe = sum(1 for f in files if f"{path_in_repo}/per_probe/" in f)
    n_local = len(list((out_dir / "per_probe").glob("*.pt")))
    if n_per_probe < expected_per_probe:
        raise RuntimeError(
            f"per-probe upload verification failed on {repo_used}: remote has "
            f"{n_per_probe} files under {path_in_repo}/per_probe/, expected >= "
            f"{expected_per_probe} (local per_probe dir has {n_local} .pt files)"
        )
    logger.info(
        "Upload verified on %s: %d files under %s (%d per-probe >= %d expected)",
        repo_used,
        len(files),
        path_in_repo,
        n_per_probe,
        expected_per_probe,
    )
    return {
        "repo": repo_used,
        "path_in_repo": path_in_repo,
        "n_files": len(files),
        "n_per_probe": n_per_probe,
    }


def upload_battery(battery_path: Path) -> None:
    """Upload battery.json to <prefix>/inputs/ (plan §8 HF-uploads row)."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(battery_path),
        path_in_repo=f"{HF_PREFIX}/inputs/battery.json",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message="issue594: battery.json input",
    )


def upload_probes_file(probes_path: Path) -> None:
    """Upload the --probes-file JSON to <prefix>/inputs/ (plan v2 §2 item 6,
    mirroring upload_battery)."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(probes_path),
        path_in_repo=f"{HF_PREFIX}/inputs/{probes_path.name}",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message="issue594: probes-file input (probe-genre-generalization)",
    )


def load_probes_file(path: Path) -> tuple[list[str], str]:
    """Load probe texts from a builder JSON; assert its OWN pool hash.

    Returns (texts, file_hash). The file's ``meta.probe_pool_hash`` must
    equal ``probes_hash`` over the ordered probe texts — fail loud on drift
    between the committed JSON and its meta block.
    """
    with open(path) as f:
        payload = json.load(f)
    texts = [p["text"] for p in payload["probes"]]
    file_hash = payload["meta"]["probe_pool_hash"]
    got = probes_hash(texts)
    assert got == file_hash, (
        f"probes-file pool hash drifted: meta says {file_hash[:16]}..., "
        f"texts hash to {got[:16]}... ({path})"
    )
    return texts, file_hash


# ── Smoke instance selection (plan §14 item 5) ──────────────────────────────


def smoke_instances(instances: list[dict]) -> list[dict]:
    """4 instances, one per structural template shape (supersedes §3's '2')."""
    by_id = {i["id"]: i for i in instances}
    persona = next(i for i in instances if i["family"] == "persona" and not i["prefix_messages"])
    icl = next(i for i in instances if i["family"] == "icl")
    wildchat = next(
        i for i in instances if i["family"] == "wildchat" and len(i["prefix_messages"]) > 2
    )
    default = by_id["f6_default_template"]
    picks = [persona, icl, wildchat, default]
    assert len({p["id"] for p in picks}) == 4, [p["id"] for p in picks]
    return picks


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #594 Phase 1: extract per-layer context vectors."
    )
    parser.add_argument("--battery", type=Path, default=BATTERY_PATH)
    parser.add_argument(
        "--probes-file",
        type=Path,
        default=None,
        help="builder JSON probe pool (plan v2 §4); default None = the Betley "
        "preregistered pool with the battery-meta hash assert (byte-for-byte "
        "parent behavior)",
    )
    parser.add_argument(
        "--hf-subdir",
        default=None,
        help="verbatim HF upload sub-directory under the issue prefix (plan v2 §4); "
        "default None = smoke_probe/analysis_tensors by --smoke; the v2 launch "
        "passes analysis_tensors_probegen",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--n-probes",
        type=int,
        default=0,
        help="cap the probe pool (0 = full pool; smoke default 4)",
    )
    parser.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="identical pipeline on 4 instances (one per template shape) x 4 probes "
        "into <out-dir>_smoke + tiny HF upload probe (plan §14 item 5)",
    )
    parser.add_argument("--no-upload", action="store_true", help="skip HF upload (local smoke)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    phase("load")
    # Bind CVD BEFORE the first CUDA allocation (the +gpu_id clobber gotcha;
    # same pattern as issue404_predictor_cossim).
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    out_dir = Path(f"{args.out_dir}_smoke") if args.smoke else args.out_dir
    per_probe_dir = out_dir / "per_probe"
    per_probe_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "extraction_manifest.json"

    payload, instances = load_battery(args.battery)
    if args.probes_file is not None:
        probes, probes_file_hash = load_probes_file(args.probes_file)
        probe_pool_source = {
            "probe_pool_source": "probes_file",
            "probes_file": str(args.probes_file),
            "probes_file_hash": probes_file_hash,
        }
        logger.info(
            "probes-file mode (%s, pool hash %s...): battery-meta Betley probe "
            "assert BYPASSED — the battery hash pins the BETLEY pool, this run "
            "deliberately swaps the probe pool (plan v2 §4)",
            args.probes_file,
            probes_file_hash[:16],
        )
    else:
        main8 = set(fetch_betley_main_8())
        probes = fetch_preregistered_probes(n=200, exclude=main8)
        assert probes_hash(probes) == payload["meta"]["probe_pool_hash"], (
            "probe pool drifted since battery build"
        )
        # No added manifest keys on the default path: plan v2 §4 requires the
        # no-flag invocation to stay byte-for-byte the parent behavior.
        probe_pool_source = {}
    n_probes_cap = args.n_probes or (4 if args.smoke else len(probes))
    probes = probes[:n_probes_cap]
    instances_to_run = smoke_instances(instances) if args.smoke else instances
    logger.info(
        "Extraction: %d instances x %d probes (smoke=%s, out=%s)",
        len(instances_to_run),
        len(probes),
        args.smoke,
        out_dir,
    )

    import wandb

    run = wandb.init(
        project="explore-persona-space",
        name="issue594-extract-smoke" if args.smoke else "issue594-extract",
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "n_instances": len(instances_to_run),
            "n_probes": len(probes),
            "smoke": args.smoke,
        },
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, (
        f"model has {n_layers} decoder layers, expected {args.expected_layers} (A1)"
    )
    assert hidden == args.expected_hidden, (
        f"model hidden_size {hidden}, expected {args.expected_hidden} (A1)"
    )

    # Record the templated default-instance prompt in the manifest (A7).
    default_preview = tokenizer.apply_chat_template(
        messages_for_instance(
            next(i for i in instances if i["id"] == "f6_default_template"), probes[0]
        ),
        tokenize=False,
        add_generation_prompt=True,
    )

    phase("extract")
    default_lens = default_template_token_lens(tokenizer, instances, probes)

    manifest: dict = {
        "instances": {},
        "probe_pool_n": len(probes),
        "probe_pool_hash": probes_hash(probes),
        **probe_pool_source,
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "smoke": args.smoke,
        "default_template_prompt_preview": default_preview[:2000],
        "default_template_token_lens": default_lens,
        "metadata": reproducibility_metadata({"script": "issue594_extract_context_vectors"}),
    }
    if manifest_path.exists():
        with open(manifest_path) as f:
            prior = json.load(f)
        if (
            prior.get("probe_pool_hash") == manifest["probe_pool_hash"]
            and prior.get("model") == args.model
        ):
            manifest["instances"] = prior.get("instances", {})
            logger.info("Resume: %d instances already complete", len(manifest["instances"]))

    capture = LayerCapture(model, n_layers)
    all_token_lens: list[int] = []
    t0 = time.time()
    try:
        for idx, instance in enumerate(instances_to_run, 1):
            iid = instance["id"]
            pp_path = per_probe_dir / f"{iid}.pt"
            if iid in manifest["instances"] and pp_path.exists():
                logger.info(
                    "[%d/%d] %s already complete; skipping", idx, len(instances_to_run), iid
                )
                continue
            t_inst = time.time()
            per_probe_fp32, token_lens = extract_for_instance(
                model, tokenizer, instance, probes, capture, n_layers
            )
            assert per_probe_fp32.shape == (len(probes), n_layers, hidden), per_probe_fp32.shape
            # Checkpoint-per-instance: per-probe fp16 + the TRUE fp32 mean +
            # manifest row land NOW. Storing the fp32 mean alongside makes the
            # A8 fp16-storage sanity check in the analysis script meaningful
            # (fp32-true-mean vs fp16-recomputed-mean) instead of vacuous.
            torch.save(
                {
                    "instance_id": iid,
                    "family": instance["family"],
                    "tensor": per_probe_fp32.to(torch.float16),
                    "mean_fp32": per_probe_fp32.mean(dim=0),
                    "probe_pool_hash": manifest["probe_pool_hash"],
                },
                pp_path,
            )
            delta_cov = statistics.median(
                t - d for t, d in zip(token_lens, default_lens, strict=True)
            )
            manifest["instances"][iid] = {
                "family": instance["family"],
                "sub_label": instance["sub_label"],
                "label": instance["label"],
                "prompt_token_lens": token_lens,
                "ctx_token_len_content": content_token_len(tokenizer, instance),
                "ctx_token_len_delta_vs_default": delta_cov,
                "position_assert_pass": True,
                "seconds": round(time.time() - t_inst, 2),
            }
            write_manifest(manifest_path, manifest)
            all_token_lens.extend(token_lens)
            run.log(
                {
                    "instances_completed": len(manifest["instances"]),
                    "ctx_token_len_content": manifest["instances"][iid]["ctx_token_len_content"],
                    "mean_prompt_tokens": sum(token_lens) / len(token_lens),
                }
            )
            logger.info(
                "[%d/%d] %s done (%.1fs, ctx_content=%d tok)",
                idx,
                len(instances_to_run),
                iid,
                time.time() - t_inst,
                manifest["instances"][iid]["ctx_token_len_content"],
            )
    finally:
        capture.remove()
    logger.info("Extraction loop done in %.1f min", (time.time() - t0) / 60)
    if all_token_lens:
        run.log({"prompt_token_len_hist": wandb.Histogram(all_token_lens)})

    # ── Assemble the probe-mean tensor (fp32, battery order) ────────────────
    phase("assemble")
    ids, families, means = [], [], []
    for instance in instances_to_run:
        blob = torch.load(per_probe_dir / f"{instance['id']}.pt", weights_only=True)
        assert blob["instance_id"] == instance["id"]
        means.append(blob["mean_fp32"])  # (L, H) TRUE fp32 mean saved at extraction time
        ids.append(instance["id"])
        families.append(instance["family"])
    mean_tensor = torch.stack(means)  # (N, L, H)
    assert mean_tensor.shape == (len(instances_to_run), n_layers, hidden), mean_tensor.shape
    torch.save(
        {
            "tensor": mean_tensor,
            "instance_ids": ids,
            "families": families,
            "probe_pool_n": len(probes),
            "probe_pool_hash": manifest["probe_pool_hash"],
            "model": args.model,
            "metadata": reproducibility_metadata({"script": "issue594_extract_context_vectors"}),
        },
        out_dir / "context_vectors_mean.pt",
    )
    logger.info("Wrote %s (%s)", out_dir / "context_vectors_mean.pt", tuple(mean_tensor.shape))

    # ── Upload + verify (plan §8; one bulk commit) ───────────────────────────
    upload_info: dict = {"skipped": True}
    if not args.no_upload:
        phase("upload")
        upload_info = upload_outputs(
            out_dir,
            smoke=args.smoke,
            expected_per_probe=len(instances_to_run),
            hf_subdir=args.hf_subdir,
        )
        # Record repo_used in the manifest itself and backfill the remote copy,
        # so Phase 2 can resolve the right repo (primary vs quota-403 overflow)
        # from the downloaded artifacts — not only from the results sentinel.
        # The manifest is a small non-LFS .json, so this backfill succeeds even
        # under the account-wide LFS quota 403 (upload-policy.md).
        manifest["upload"] = upload_info
        write_manifest(manifest_path, manifest)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{upload_info['path_in_repo']}/extraction_manifest.json",
            repo_id=upload_info["repo"],
            repo_type="dataset",
            commit_message="issue594: manifest upload-provenance backfill (repo_used)",
        )
        if not args.smoke:
            upload_battery(args.battery)
            if args.probes_file is not None:
                upload_probes_file(args.probes_file)
    else:
        manifest["upload"] = upload_info
        write_manifest(manifest_path, manifest)

    note = (
        f"issue594 extraction {'SMOKE ' if args.smoke else ''}complete: "
        f"{len(manifest['instances'])} instances x {len(probes)} probes, "
        f"layers={n_layers}, hidden={hidden}, upload={upload_info}"
    )
    write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    run.finish()
    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
