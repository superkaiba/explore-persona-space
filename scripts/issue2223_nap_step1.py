"""Issue #2223 NAP round — committed step-1 wrapper over the paper pipeline.

Runs the assistant-axis paper's step-1 role-response generation
(``external/assistant-axis`` ``RoleResponseGenerator``) with three additions
the external checkout cannot carry (it is git-UNTRACKED — pods obtain it via
``scripts/issue2203_pod_bootstrap_engine.sh`` at the pinned SHA):

1. **In-process role sharding** (``--shard-id/--num-shards``): the paper's own
   ``pipeline/1_generate.py`` multi-worker path uses mp spawn workers that
   re-import vLLM unpatched; this wrapper shards the sorted role list i::n
   in-process so one wrapper == one vLLM engine == one CVD-pinned GPU.
2. **vLLM hang-mitigation env pins**: ``EPM_VLLM_ENFORCE_EAGER=1`` /
   ``EPM_VLLM_DISABLE_PREFIX_CACHING=1`` are honored by patching ``vllm.LLM``
   in the module namespace BEFORE ``VLLMGenerator.load()`` runs its
   call-time ``from vllm import LLM`` (the external code exposes no engine
   kwarg seam).
3. **Atomic per-role saves + regime-fingerprinted resume**: the external
   ``save_responses`` is non-atomic, so a killed worker would leave a partial
   JSONL that skip-existing would treat as complete; this wrapper writes
   tmp + ``os.replace`` and pins the decode regime in ``step1_regime.json``.

Content hygiene: this wrapper never prints response text — logs carry role
names, counts, and paths only (the external module's own logging does the
same). Row format is byte-compatible with the paper pipeline: per-role
``<role>.jsonl`` rows ``{system_prompt, prompt_index, question_index,
question, conversation, label}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# vLLM v1 EngineCore dies silently under fork() when the parent touched
# transformers/tokenizer helpers pre-LLM() (gotchas.md, #628) — set BEFORE
# any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_nap_step1.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from scripts.issue2223_casestudy_replay import _atomic_write_json, _log  # noqa: E402


def assistant_axis_dir() -> Path:
    """The external assistant-axis checkout (env-overridable for VM-side checks)."""
    override = os.environ.get("EPS_ASSISTANT_AXIS_DIR")
    return Path(override) if override else REPO / "external" / "assistant-axis"


def _load_assistant_axis_submodule(name: str):
    """Import ``assistant_axis.<name>`` WITHOUT executing the package __init__.

    The real ``assistant_axis/__init__.py`` imports plotly/sklearn (absent on
    pods) — the established bypass is the paper_engine.py precedent
    (``src/explore_persona_space/experiments/issue2203/paper_engine.py``):
    load submodules standalone. Here a stub package module with ``__path__``
    pointed at the real package dir lets normal submodule import machinery —
    including ``generation.py``'s deferred ``from .models import get_config`` —
    resolve against the real files while the plotly-bearing __init__ never runs.
    """
    import importlib
    import types

    ext = assistant_axis_dir()
    pkg_dir = ext / "assistant_axis"
    assert (pkg_dir / "generation.py").exists(), (
        f"external assistant-axis checkout absent at {ext} — run "
        "scripts/issue2203_pod_bootstrap_engine.sh (pod) or set EPS_ASSISTANT_AXIS_DIR"
    )
    if "assistant_axis" not in sys.modules:
        stub = types.ModuleType("assistant_axis")
        stub.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        stub._nap_init_bypass = True  # type: ignore[attr-defined]
        sys.modules["assistant_axis"] = stub
    return importlib.import_module(f"assistant_axis.{name}")


def _patch_vllm_llm() -> None:
    """Honor EPM_VLLM_ENFORCE_EAGER / EPM_VLLM_DISABLE_PREFIX_CACHING.

    ``VLLMGenerator.load()`` does ``from vllm import LLM`` at CALL time with no
    engine-kwarg seam, so rebinding ``vllm.LLM`` here (before ``load()``)
    injects the mitigation kwargs. setdefault-only — an explicit caller kwarg
    would win; idempotent across calls.
    """
    import vllm

    orig = vllm.LLM
    if getattr(orig, "_nap_patched", False):
        return

    def patched_llm(*args, **kwargs):
        if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
            kwargs.setdefault("enforce_eager", True)
        if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
            kwargs.setdefault("enable_prefix_caching", False)
        _log(
            "[nap-step1] vllm.LLM engine kwargs: "
            f"enforce_eager={kwargs.get('enforce_eager')} "
            f"enable_prefix_caching={kwargs.get('enable_prefix_caching')}"
        )
        return orig(*args, **kwargs)

    patched_llm._nap_patched = True  # type: ignore[attr-defined]
    vllm.LLM = patched_llm


def _regime(args) -> dict:
    from scripts import issue2203_common as C

    return C.regime_fingerprint(
        round_label="native_axis_fidelity_preimage",
        phase="step1",
        model=args.model,
        question_count=int(args.question_count),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        max_model_len=int(args.max_model_len),
        prompt_indices=list(range(5)),
    )


def run(args) -> int:
    from scripts import issue2203_common as C

    _patch_vllm_llm()
    RoleResponseGenerator = _load_assistant_axis_submodule("generation").RoleResponseGenerator

    ext = assistant_axis_dir()
    # Pinned-checkout layout (a98961956): roles at data/roles/instructions/
    # (276 role JSONs incl. assistant.json + default.json), questions at
    # data/extraction_questions.jsonl — the upstream pipeline/1_generate.py
    # argparse defaults. data/prompts/ does NOT exist in the checkout.
    roles_dir = Path(args.roles_dir) if args.roles_dir else ext / "data" / "roles" / "instructions"
    questions_file = (
        Path(args.questions_file)
        if args.questions_file
        else ext / "data" / "extraction_questions.jsonl"
    )
    assert questions_file.exists(), f"questions file absent: {questions_file}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime = _regime(args)
    regime_path = out_dir / "step1_regime.json"
    if regime_path.exists():
        C.check_regime(json.loads(regime_path.read_text()), regime, regime_path)
    else:
        _atomic_write_json(regime_path, regime)

    role_files = sorted(roles_dir.glob("*.json"))
    assert role_files, f"no role JSONs under {roles_dir}"
    if args.roles_dir is None:
        # Layout guard for the DEFAULT dir only (an explicit --roles-dir may be
        # a deliberate subset): the pinned checkout carries exactly 276 role
        # JSONs — any other count means a wrong/stale external checkout.
        assert len(role_files) == 276, (
            f"pinned assistant-axis checkout should carry 276 role JSONs under "
            f"{roles_dir}, found {len(role_files)} — wrong/stale external checkout "
            "(re-run scripts/issue2203_pod_bootstrap_engine.sh)"
        )
    if args.roles:
        wanted = {r.strip() for r in args.roles.split(",") if r.strip()}
        role_files = [f for f in role_files if f.stem in wanted]
        missing = wanted - {f.stem for f in role_files}
        assert not missing, f"--roles names not found under {roles_dir}: {sorted(missing)}"
    n_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    assert 0 <= shard_id < n_shards, (shard_id, n_shards)
    my_files = role_files[shard_id::n_shards]

    gen = RoleResponseGenerator(
        model_name=args.model,
        roles_dir=str(roles_dir),
        output_dir=str(out_dir),
        questions_file=str(questions_file),
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        question_count=int(args.question_count),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_p=float(args.top_p),
    )

    pending = [f for f in my_files if not (out_dir / f"{f.stem}.jsonl").exists()]
    _log(
        f"[nap-step1] shard {shard_id}/{n_shards}: {len(my_files)} roles, "
        f"{len(my_files) - len(pending)} done, {len(pending)} pending"
    )
    if not pending:
        return 0

    t0 = time.time()
    n_skipped_norole = 0
    for k, role_file in enumerate(pending):
        role_name = role_file.stem
        role_data = gen.load_role(role_file)
        if "instruction" not in role_data:
            _log(f"[nap-step1] WARNING: {role_name} lacks 'instruction' — skipped")
            n_skipped_norole += 1
            continue
        rows = gen.generate_role_responses(role_name, role_data)
        assert rows, f"role {role_name} produced 0 rows"
        out = out_dir / f"{role_name}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        os.replace(tmp, out)
        _log(
            f"[nap-step1] unit {k + 1}/{len(pending)} {role_name} rows={len(rows)} "
            f"elapsed={time.time() - t0:.0f}s"
        )
    _log(
        f"[nap-step1] shard {shard_id} DONE: {len(pending) - n_skipped_norole} roles written, "
        f"{n_skipped_norole} skipped (no instruction), {time.time() - t0:.0f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--output-dir", required=False, help="per-role JSONL destination")
    ap.add_argument("--roles-dir", default=None, help="default: <external>/data/roles/instructions")
    ap.add_argument(
        "--questions-file",
        default=None,
        help="default: <external>/data/extraction_questions.jsonl",
    )
    ap.add_argument("--roles", default=None, help="comma list of role stems (smoke slices)")
    ap.add_argument(
        "--question-count", type=int, default=40, help="questions per role (this round: 40)"
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from scripts import issue2203_common as C  # noqa: F401

        ext = assistant_axis_dir()
        if (ext / "assistant_axis" / "generation.py").exists():
            gen_mod = _load_assistant_axis_submodule("generation")
            models_mod = _load_assistant_axis_submodule("models")
            assert hasattr(gen_mod, "RoleResponseGenerator"), ext
            assert hasattr(models_mod, "get_config"), ext
            ext_state = f"external imports resolved from {ext}"
        else:
            ext_state = (
                f"external/ absent at {ext} — pod bootstrap provides it "
                "(assistant_axis imports NOT resolved here)"
            )
        assert_args_attributes_defined(__file__)
        print(f"[import-check] ok ({ext_state})")
        return 0
    assert args.output_dir, "--output-dir is required (or --import-check)"
    rc = run(args)
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
