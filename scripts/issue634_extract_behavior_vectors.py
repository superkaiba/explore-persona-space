#!/usr/bin/env python3
"""Issue #634 Phase 1: extract 28-layer behavior (persona) vectors for 275 roles.

Method A (last-input-token absolute mean), the read slot BYTE-IDENTICAL to
#594's context vectors: ``model.model.layers[i]`` output at the last input
token under ``add_generation_prompt=True`` (the newline of
``<|im_start|>assistant\\n``), all 28 decoder layers (pre-final-norm residual,
NOT ``output_hidden_states``). The vector definition is unchanged from
``extract_persona_vectors.py::extract_method_a``; this wrapper fills in the
ALL-28-layer coverage + the #594 manifest schema + the per-forward position
assert (which the bare ``extract_method_a`` does not carry).

For each of the 275 roles: 5 system prompts x K=40 of the 240 shared
extraction questions (the SAME 40 question indices for every role — sampled
ONCE globally with seed 42, NOT per role). The role centroid is the mean over
the 5 x 40 = 200 forward-pass last-token vectors per layer.

Outputs (``--output-dir``, checkpoint-per-role with resume-skip):

- ``per_role/<role>.pt``       fp32 (28, 3584) per-role centroid + provenance
- ``behavior_vectors_mean.pt`` fp32 (275, 28, 3584) tensor + index + manifest schema
- ``extraction_manifest.json`` per-role provenance + repro metadata

Manifest mirrors #594's schema: ``instance_ids`` (= role names), ``families``
(from the frozen Panel-B map; ``None`` for roles not in Panel B), ``model``,
``n_questions=40``, ``n_prompts=5``, ``seed=42``, ``sampled_question_indices``,
``sampled_question_indices_hash`` (the K=40 sample fingerprint), repro metadata.

In-process asserts: ``n_layers == 28``, ``hidden == 3584``, per-forward
last-3-token decode == ``<|im_start|>assistant\\n`` (the #594 position assert,
copied verbatim).

Smoke gate (option C): ``--smoke-roles N`` runs only the first N roles (default
0 = all). The orchestrator runs ``--smoke-roles 1`` first to time the actual
GCP A100-80 lane before the full 275-role run; if extrapolated cost > 20 GPU-h,
descope K (40 -> 20) via ``--n-questions 20``.

Pod-side contract: emits ``[phase=...]`` log lines ending in ``[phase=done]``
and writes a ``poll_pipeline.py``-conformant end-of-run sentinel. Pod-side code
NEVER shells out to task.py.

Usage::

    # Full pod run (GCP lora-7b A100-80 lane):
    uv run python scripts/issue634_extract_behavior_vectors.py \\
        --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \\
        --n-questions 40 --seed 42 --gpu-id 0 \\
        --output-dir data/issue634/behavior_vectors

    # 1-role smoke gate (option C, times the A100 lane):
    uv run python scripts/issue634_extract_behavior_vectors.py \\
        --smoke-roles 1 --output-dir data/issue634/behavior_vectors_smoke

    # local CPU micro-smoke (tiny random Qwen2 substitute, no upload):
    uv run python scripts/issue634_extract_behavior_vectors.py \\
        --smoke-roles 1 --n-questions 2 --layers 0 --device cpu --no-upload \\
        --smoke-cpu --wandb-mode disabled --data-dir data/assistant_axis \\
        --output-dir /tmp/issue634_cpu_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_common import (  # noqa: E402
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
)

load_dotenv()

logger = logging.getLogger("issue634_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GENERATION_SUFFIX = "<|im_start|>assistant\n"
SENTINEL_SCHEMA_VERSION = 1
TASK_ID = 634
HF_PREFIX = "issue634_behavior_geometry"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "assistant_axis"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "issue634" / "behavior_vectors"
FAMILY_MAP_PATH = PROJECT_ROOT / "data" / "issue634" / "behavior_family_map.json"
WANDB_PROJECT = "issue634"
ALL_LAYERS = list(range(EXPECTED_LAYERS))


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line (PHASE_RE on the log tail)."""
    print(f"[phase={name}]", flush=True)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_roles(data_dir: Path, roles_filter: list[str] | None = None) -> dict[str, list[str]]:
    """Load role -> list of 5 system prompts from the instruction files.

    Mirrors ``extract_persona_vectors.load_roles`` but parameterized on the
    data dir (so the local smoke can point at the repo-root copy when the
    worktree is sparse). Returns roles in sorted order.
    """
    with open(data_dir / "role_list.json") as f:
        all_roles = json.load(f)
    if roles_filter:
        all_roles = {k: v for k, v in all_roles.items() if k in roles_filter}
    role_prompts: dict[str, list[str]] = {}
    instructions_dir = data_dir / "instructions"
    for role_name in sorted(all_roles.keys()):
        instr_path = instructions_dir / f"{role_name}.json"
        if not instr_path.exists():
            logger.warning("No instruction file for %s, skipping", role_name)
            continue
        with open(instr_path) as f:
            data = json.load(f)
        role_prompts[role_name] = [item["pos"] for item in data["instruction"]]
    return role_prompts


def sample_question_indices(all_questions: list[str], k: int, seed: int) -> list[int]:
    """Sample k of the 240 questions ONCE globally (same set for every role).

    Deterministic seed-42 sample WITHOUT replacement, returned sorted so the
    selection is stable and reproducible. k >= len -> all indices.
    """
    import random

    n = len(all_questions)
    if k >= n:
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def load_questions(data_dir: Path) -> list[str]:
    """Load the 240 shared extraction questions (ordered)."""
    qs: list[str] = []
    with open(data_dir / "extraction_questions.jsonl") as f:
        for line in f:
            qs.append(json.loads(line)["question"])
    return qs


def indices_hash(indices: list[int]) -> str:
    """Stable sha256 over the sampled question indices (provenance pin)."""
    h = hashlib.sha256()
    h.update(json.dumps(indices, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def load_family_map(path: Path) -> dict[str, str]:
    """Load the frozen Panel-B role->family map (Phase 0 output); {} if absent."""
    if not path.exists():
        logger.warning("family map %s absent; manifest families will be all None", path)
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("map", payload)  # tolerate a bare map or the wrapped payload


# ── Hook capture (copied from issue594_extract_context_vectors.LayerCapture) ──


class LayerCapture:
    """Forward hooks on every decoder block; keeps the latest (1, T, H) per layer."""

    def __init__(self, model, layers: list[int]):
        self.latest: dict[int, torch.Tensor] = {}
        self._handles = []
        for li in layers:
            self._handles.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.latest[layer_idx] = hs.detach()

        return hook_fn

    def last_token_stack(self, layers: list[int]) -> torch.Tensor:
        """(L, H) fp32 CPU stack of the last-position activation per layer."""
        vecs = [self.latest[li][0, -1, :].float().cpu() for li in layers]
        self.latest.clear()
        return torch.stack(vecs)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def build_chat_text(tokenizer, system_prompt: str, question: str) -> str:
    """Tokenizer-formatted chat text for one (system, user) pair."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_role(
    model,
    tokenizer,
    prompts: list[str],
    questions: list[str],
    capture: LayerCapture,
    layers: list[int],
) -> tuple[torch.Tensor, int]:
    """Forward every (prompt, question) under the role; mean last-token vectors.

    Returns (centroid fp32 (L, H), n_forwards). Per-forward position assert:
    the last 3 input tokens must decode to the assistant-header suffix — fail
    LOUD on any drift (the #594 control, copied verbatim).
    """
    per_forward: list[torch.Tensor] = []
    for sys_prompt in prompts:
        for q in questions:
            text = build_chat_text(tokenizer, sys_prompt, q)
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
            suffix = tokenizer.decode(inputs["input_ids"][0, -3:])
            assert suffix == GENERATION_SUFFIX, (
                f"position assert failed (prompt={sys_prompt[:30]!r} q={q[:30]!r}): "
                f"last-3-token decode {suffix!r} != {GENERATION_SUFFIX!r}"
            )
            with torch.no_grad():
                _ = model(**inputs)
            per_forward.append(capture.last_token_stack(layers))  # (L, H)
    stacked = torch.stack(per_forward)  # (n_forwards, L, H)
    centroid = stacked.mean(dim=0)  # (L, H)
    return centroid, len(per_forward)


# ── Manifest / sentinel ──────────────────────────────────────────────────────


def write_manifest(path: Path, manifest: dict) -> None:
    """Atomic-ish manifest rewrite (tmp + rename) after every role."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


def write_sentinel(kind: str, note: str) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{TASK_ID}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": TASK_ID,
        "by": "issue634_extract_behavior_vectors",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


# ── HF upload (mirrors issue594 upload_outputs) ──────────────────────────────


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def upload_outputs(out_dir: Path, sub: str, expected_per_role: int) -> dict:
    """Bulk-upload the extraction outputs to the HF data repo and verify.

    ONE ``upload_folder`` commit (well under the 256/hr cap), verified via
    ``huggingface_hub.list_repo_files`` (never the ``hf`` CLI). On the
    account-wide LFS storage-quota 403, falls back to the private overflow repo
    per .claude/rules/upload-policy.md. Fail-loud otherwise.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    path_in_repo = f"{HF_PREFIX}/{sub}"
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message="issue634: behavior vectors upload",
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
            commit_message="issue634: behavior vectors upload (quota-403 overflow fallback)",
        )
    files = [
        f for f in api.list_repo_files(repo_used, repo_type="dataset") if f.startswith(path_in_repo)
    ]
    expected = {
        f"{path_in_repo}/behavior_vectors_mean.pt",
        f"{path_in_repo}/extraction_manifest.json",
    }
    missing = expected - set(files)
    if missing:
        raise RuntimeError(f"upload verification failed; missing on {repo_used}: {missing}")
    n_per_role = sum(1 for f in files if f"{path_in_repo}/per_role/" in f)
    n_local = len(list((out_dir / "per_role").glob("*.pt")))
    if n_per_role < expected_per_role:
        raise RuntimeError(
            f"per-role upload verification failed on {repo_used}: remote has "
            f"{n_per_role} files under {path_in_repo}/per_role/, expected >= "
            f"{expected_per_role} (local per_role dir has {n_local} .pt files)"
        )
    logger.info(
        "Upload verified on %s: %d files under %s (%d per-role >= %d expected)",
        repo_used,
        len(files),
        path_in_repo,
        n_per_role,
        expected_per_role,
    )
    return {
        "repo": repo_used,
        "path_in_repo": path_in_repo,
        "n_files": len(files),
        "n_per_role": n_per_role,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #634 Phase 1: extract 28-layer behavior vectors (Method A)."
    )
    parser.add_argument("--layers", type=int, nargs="+", default=ALL_LAYERS)
    parser.add_argument("--n-questions", type=int, default=40, help="K of the 240 (seed-42 sample)")
    parser.add_argument("--n-prompts", type=int, default=5, help="system prompts per role (max 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--family-map", type=Path, default=FAMILY_MAP_PATH)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--smoke-roles",
        type=int,
        default=0,
        help="run only the first N roles (0 = all 275); the option-C smoke gate",
    )
    parser.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="local micro-smoke: skip the n_layers/hidden assert vs 28/3584 (tiny stub model)",
    )
    parser.add_argument(
        "--expected-layers", type=int, default=EXPECTED_LAYERS, help="model-shape assert"
    )
    parser.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    parser.add_argument("--no-upload", action="store_true", help="skip HF upload (local smoke)")
    parser.add_argument("--hf-subdir", default="analysis_tensors", help="HF upload sub-directory")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    phase("load")
    # Bind CVD BEFORE the first CUDA allocation (the +gpu_id clobber gotcha).
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    layers = sorted(args.layers)
    out_dir = args.output_dir
    per_role_dir = out_dir / "per_role"
    per_role_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "extraction_manifest.json"

    role_prompts = load_roles(args.data_dir)
    role_names = sorted(role_prompts.keys())
    if args.smoke_roles > 0:
        role_names = role_names[: args.smoke_roles]
        role_prompts = {r: role_prompts[r] for r in role_names}
    all_questions = load_questions(args.data_dir)
    q_idx = sample_question_indices(all_questions, args.n_questions, args.seed)
    questions = [all_questions[i] for i in q_idx]
    family_map = load_family_map(args.family_map)
    logger.info(
        "Extraction: %d roles x %d prompts x %d questions (seed %d), %d layers, out=%s",
        len(role_names),
        args.n_prompts,
        len(questions),
        args.seed,
        len(layers),
        out_dir,
    )

    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"issue634-extract{'-smoke' if args.smoke_roles else ''}",
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "n_roles": len(role_names),
            "n_prompts": args.n_prompts,
            "n_questions": len(questions),
            "n_layers": len(layers),
            "seed": args.seed,
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

    n_layers_model = len(model.model.layers)
    hidden = model.config.hidden_size
    if not args.smoke_cpu:
        assert n_layers_model == args.expected_layers, (
            f"model has {n_layers_model} decoder layers, expected {args.expected_layers}"
        )
        assert hidden == args.expected_hidden, (
            f"model hidden_size {hidden}, expected {args.expected_hidden}"
        )
    assert max(layers) < n_layers_model, (
        f"requested layer {max(layers)} >= model layer count {n_layers_model}"
    )

    phase("extract")
    manifest: dict = {
        "instance_ids": [],
        "families": [],
        "model": args.model,
        "n_layers": len(layers),
        "layers": layers,
        "hidden": hidden,
        "n_prompts": args.n_prompts,
        "n_questions": len(questions),
        "seed": args.seed,
        "sampled_question_indices": q_idx,
        "sampled_question_indices_hash": indices_hash(q_idx),
        "family_map_path": str(args.family_map),
        "roles": {},
        "metadata": reproducibility_metadata({"script": "issue634_extract_behavior_vectors"}),
    }
    # Resume: reuse the prior manifest's per-role rows when the sample matches.
    if manifest_path.exists():
        with open(manifest_path) as f:
            prior = json.load(f)
        if (
            prior.get("sampled_question_indices_hash") == manifest["sampled_question_indices_hash"]
            and prior.get("model") == args.model
            and prior.get("layers") == layers
        ):
            manifest["roles"] = prior.get("roles", {})
            logger.info("Resume: %d roles already complete", len(manifest["roles"]))

    capture = LayerCapture(model, layers)
    t0 = time.time()
    try:
        for idx, role in enumerate(role_names, 1):
            pr_path = per_role_dir / f"{role}.pt"
            if role in manifest["roles"] and pr_path.exists():
                logger.info("[%d/%d] %s already complete; skipping", idx, len(role_names), role)
                continue
            t_role = time.time()
            prompts = role_prompts[role][: args.n_prompts]
            centroid, n_fwd = extract_role(model, tokenizer, prompts, questions, capture, layers)
            assert centroid.shape == (len(layers), hidden), centroid.shape
            torch.save(
                {
                    "role": role,
                    "family": family_map.get(role),
                    "tensor": centroid,  # (L, H) fp32 true mean
                    "n_prompts": len(prompts),
                    "n_questions": len(questions),
                    "sampled_question_indices_hash": manifest["sampled_question_indices_hash"],
                },
                pr_path,
            )
            manifest["roles"][role] = {
                "family": family_map.get(role),
                "n_forwards": n_fwd,
                "seconds": round(time.time() - t_role, 2),
                "position_assert_pass": True,
            }
            write_manifest(manifest_path, manifest)
            run.log(
                {"roles_completed": len(manifest["roles"]), "role_seconds": time.time() - t_role}
            )
            logger.info(
                "[%d/%d] %s done (%.1fs, %d forwards)",
                idx,
                len(role_names),
                role,
                time.time() - t_role,
                n_fwd,
            )
    finally:
        capture.remove()
    logger.info("Extraction loop done in %.1f min", (time.time() - t0) / 60)

    # ── Assemble the role-mean tensor (fp32, sorted-role order) ──────────────
    phase("assemble")
    ids, families, means = [], [], []
    for role in role_names:
        blob = torch.load(per_role_dir / f"{role}.pt", weights_only=True)
        assert blob["role"] == role
        assert blob["tensor"].shape == (len(layers), hidden), blob["tensor"].shape
        means.append(blob["tensor"])
        ids.append(role)
        families.append(family_map.get(role))
    mean_tensor = torch.stack(means)  # (N, L, H)
    assert mean_tensor.shape == (len(role_names), len(layers), hidden), mean_tensor.shape
    manifest["instance_ids"] = ids
    manifest["families"] = families
    torch.save(
        {
            "tensor": mean_tensor,
            "instance_ids": ids,
            "families": families,
            "model": args.model,
            "n_prompts": args.n_prompts,
            "n_questions": len(questions),
            "seed": args.seed,
            "layers": layers,
            "sampled_question_indices": q_idx,
            "sampled_question_indices_hash": manifest["sampled_question_indices_hash"],
            "metadata": reproducibility_metadata({"script": "issue634_extract_behavior_vectors"}),
        },
        out_dir / "behavior_vectors_mean.pt",
    )
    write_manifest(manifest_path, manifest)
    logger.info("Wrote %s (%s)", out_dir / "behavior_vectors_mean.pt", tuple(mean_tensor.shape))

    # ── Upload + verify (one bulk commit) ────────────────────────────────────
    upload_info: dict = {"skipped": True}
    if not args.no_upload:
        phase("upload")
        upload_info = upload_outputs(out_dir, args.hf_subdir, expected_per_role=len(role_names))
        manifest["upload"] = upload_info
        write_manifest(manifest_path, manifest)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{upload_info['path_in_repo']}/extraction_manifest.json",
            repo_id=upload_info["repo"],
            repo_type="dataset",
            commit_message="issue634: manifest upload-provenance backfill",
        )
    else:
        manifest["upload"] = upload_info
        write_manifest(manifest_path, manifest)

    note = (
        f"issue634 behavior extraction {'SMOKE ' if args.smoke_roles else ''}complete: "
        f"{len(manifest['roles'])} roles x {args.n_prompts} prompts x {len(questions)} questions, "
        f"layers={len(layers)}, hidden={hidden}, upload={upload_info}"
    )
    write_sentinel("epm:smoke-result" if args.smoke_roles else "epm:results", note)
    run.finish()
    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
