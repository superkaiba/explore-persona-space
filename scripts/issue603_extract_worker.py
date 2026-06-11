#!/usr/bin/env python3
"""#603 extraction worker — ONE (family, source, seed) cell on ONE GPU.

Loads the frozen base model + the cell's LoRA adapter
(``merge_and_unload``, rig-verbatim — the #551 reproduction-gate path),
runs ``extract_per_context_shifts`` over the family's 24-persona panel x
20 probes (``arm="em"``, ``--variant`` ``same``/``base`` [default
``same`` = parent behavior; ``base`` = #603 base-text-extraction
follow-up], layers {7,14,21}, primary 14, mean-over-response ON), and
writes:

- ``<out>``: the ``.pt`` payload ``{"shifts", "manifest"}`` (#551
  schema v2 + #603 cell fields; the manifest records the REALIZED
  ``variant`` plus the per-cell truncation rate — kept responses whose
  generated length hit the ``--max-new-tokens`` cap, plan v2
  ride-along), atomic tmp+rename;
- ``<out>.manifest.json`` sidecar (grepability);
- ``<responses-out>``: per-(persona, question) generated response token
  ids + decoded texts (guard B instrumentation — plan #603 critique
  round 1 Must-Fix; the inherited rig discards them).

Adapter resolution: ``list_repo_files`` + per-file ``hf_hub_download``
(NEVER ``snapshot_download(allow_patterns=...)`` — silently returns 0
files for prefixes in the truncated siblings tail on large repos).

Invoked by ``scripts/issue603_extract_dispatch.py`` as a subprocess with
``CUDA_VISIBLE_DEVICES`` set; also runnable standalone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from _bootstrap import bootstrap

logger = bootstrap(log_name="i603_extract_worker")

from explore_persona_space.analysis.activation_shift import (  # noqa: E402
    extract_per_context_shifts,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = (7, 14, 21)
PRIMARY_LAYER = 14
MAX_NEW_TOKENS = 512


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_adapter(repo_id: str, subfolder: str) -> Path:
    """Download every file under ``subfolder`` in ``repo_id``; return the local dir.

    Uses ``list_repo_files`` + per-file ``hf_hub_download`` (the
    snapshot_download allow_patterns siblings-truncation trap on large
    repos returns 0 files silently). Fail-loud on an empty listing.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = subfolder.rstrip("/") + "/"
    files = [f for f in list_repo_files(repo_id, repo_type="model") if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"no files under {repo_id}/{prefix} — adapter missing on Hub")
    local_paths = [
        Path(hf_hub_download(repo_id=repo_id, filename=f, repo_type="model")) for f in files
    ]
    adapter_dir = {p.parent for p in local_paths}
    assert len(adapter_dir) == 1, f"adapter files resolved to multiple dirs: {adapter_dir}"
    local_dir = adapter_dir.pop()
    cfg_path = local_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{repo_id}/{prefix}: no adapter_config.json among {files}")
    return local_dir


def _assert_adapter_base(adapter_dir: Path, base_model_id: str) -> dict:
    """Plan A1: the adapter's recorded base model must match the load target."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    recorded = cfg.get("base_model_name_or_path", "")
    # Pod-trained adapters sometimes record a local path; accept when the
    # repo id is a suffix component, fail otherwise (fail-fast, no silent
    # cross-base merges).
    if recorded != base_model_id and base_model_id.split("/")[-1] not in recorded:
        raise AssertionError(
            f"adapter base model mismatch: adapter_config says {recorded!r}, "
            f"worker loads {base_model_id!r}"
        )
    return cfg


def _load_model(path_or_hub_id: str, adapter_path: str | None = None):
    """Rig-verbatim model loader (activation_shift._load_model)."""
    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_hub_id, dtype=torch.bfloat16, **model_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_hub_id, torch_dtype=torch.bfloat16, **model_kwargs
        )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def main() -> int:
    """Extract one cell's shifts + responses; write .pt + manifest + sidecar."""
    ap = argparse.ArgumentParser(description="#603 single-cell shift extraction")
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--family", required=True, choices=["fact", "refusal", "em"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-subfolder", required=True)
    ap.add_argument("--inputs-json", required=True, help="The frozen {family}_panel.json.")
    ap.add_argument("--out", required=True, help="Output .pt path.")
    ap.add_argument("--responses-out", required=True, help="Responses sidecar JSON path.")
    ap.add_argument(
        "--variant",
        default="same",
        choices=["same", "base"],
        help="Extraction text variant: 'same' = trained model's own greedy text (parent), "
        "'base' = frozen base model's greedy text (#603 base-text-extraction follow-up). "
        "Both models are teacher-forced on the identical sequence either way.",
    )
    ap.add_argument("--base-model-id", default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument(
        "--n-personas", type=int, default=0, help="Smoke: first N panel personas (+ source). 0=all."
    )
    ap.add_argument("--n-questions", type=int, default=0, help="Smoke: first N probes. 0=all.")
    args = ap.parse_args()

    inputs = json.loads(Path(args.inputs_json).read_text())
    assert inputs["family"] == args.family, (inputs["family"], args.family)
    panel: dict[str, str | None] = inputs["panel"]
    probes: list[str] = inputs["probes"]
    assert args.source in panel, f"source {args.source!r} not in panel"

    personas = dict(panel)
    if args.n_personas > 0:
        names = list(panel)[: args.n_personas]
        if args.source not in names:
            names[-1] = args.source  # the smoke subset MUST include the source
        personas = {n: panel[n] for n in names}
    questions = probes[: args.n_questions] if args.n_questions > 0 else probes

    # `no_system` (fact panel) carries a None prompt; the rig's
    # _build_chatml_prompt omits the system turn for None (the #444/#541
    # panel convention). Only `no_system` may be None — fail loud otherwise.
    none_prompts = [n for n, p in personas.items() if p is None]
    if none_prompts and none_prompts != ["no_system"]:
        raise AssertionError(f"unexpected personas with None system prompt: {none_prompts}")

    logger.info(
        "[phase=cell_load_models] cell=%s family=%s source=%s seed=%d variant=%s "
        "n_personas=%d n_questions=%d layers=%s",
        args.cell_id,
        args.family,
        args.source,
        args.seed,
        args.variant,
        len(personas),
        len(questions),
        args.layers,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)

    adapter_dir = _resolve_adapter(args.adapter_repo, args.adapter_subfolder)
    adapter_cfg = _assert_adapter_base(adapter_dir, args.base_model_id)

    base_model = _load_model(args.base_model_id, adapter_path=None)
    trained_model = _load_model(args.base_model_id, adapter_path=str(adapter_dir))

    logger.info("[phase=cell_extract] cell=%s", args.cell_id)
    t0 = time.time()
    response_sink: dict[str, list[dict[str, object]]] = {}
    shifts = extract_per_context_shifts(
        base_model=base_model,
        trained_model=trained_model,
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        arm="em",  # no marker stripping for any #603 family (plan §4 step 5)
        variant=args.variant,
        layers=args.layers,
        primary_layer=args.primary_layer,
        max_new_tokens=args.max_new_tokens,
        also_compute_mean_over_response=True,
        response_sink=response_sink,
    )
    wall_s = time.time() - t0

    # Per-cell truncation rate (plan v2 ride-along): kept responses whose
    # generated length hit the --max-new-tokens cap. Base refusal/EM
    # responses run longer than the trained templates, and a truncated
    # end-slot read sits mid-sentence — reported alongside the norm-floor
    # separator downstream (issue603_decompose.py per_cell passthrough).
    kept_records = [r for recs in response_sink.values() for r in recs if r.get("kept")]
    n_truncated = sum(1 for r in kept_records if len(r["response_ids"]) >= args.max_new_tokens)
    truncation_rate = (n_truncated / len(kept_records)) if kept_records else None

    manifest = {
        "issue": 603,
        "schema_version": 2,
        "cell_id": args.cell_id,
        "family": args.family,
        "source": args.source,
        "seed": args.seed,
        "arm": "em",
        "variant": args.variant,  # REALIZED variant — the dispatcher's resume guard keys on it
        "layer": args.primary_layer,
        "layers": list(args.layers),
        "base_model_id": args.base_model_id,
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "adapter_config_base_model": adapter_cfg.get("base_model_name_or_path"),
        "adapter_lora_r": adapter_cfg.get("r"),
        "n_personas": len(personas),
        "persona_names": list(personas.keys()),
        "n_questions": len(questions),
        "probe_sha256": inputs["probe_sha256"],
        "max_new_tokens": args.max_new_tokens,
        "n_responses_kept_total": len(kept_records),
        "n_truncated_responses": n_truncated,
        "truncation_rate": truncation_rate,
        "wall_seconds": round(wall_s, 1),
        "git_commit": _git_commit(),
        "env_versions": {
            pkg: __import__("importlib.metadata", fromlist=["version"]).version(pkg)
            for pkg in ("torch", "transformers", "peft")
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save({"shifts": shifts, "manifest": manifest}, tmp)
    tmp.rename(out_path)
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    resp_path = Path(args.responses_out)
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    with resp_path.open("w") as f:
        json.dump(
            {"cell_id": args.cell_id, "manifest": manifest, "responses": response_sink},
            f,
            ensure_ascii=False,
        )

    n_kept = {p: int(shifts[p]["n_questions_kept"]) for p in shifts}
    logger.info(
        "[phase=cell_complete] cell=%s variant=%s wall=%.1fs personas=%d min_kept=%d "
        "truncation_rate=%s (%d/%d at the %d-token cap) wrote %s + %s",
        args.cell_id,
        args.variant,
        wall_s,
        len(shifts),
        min(n_kept.values()),
        "n/a" if truncation_rate is None else f"{truncation_rate:.3f}",
        n_truncated,
        len(kept_records),
        args.max_new_tokens,
        out_path,
        resp_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
