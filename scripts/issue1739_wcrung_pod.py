"""GPU leg of the issue #1739 "wildchat_rung" (path C): random held-out WildChat.

The rung adds ONE eval-split rung (``rung="wildchat_rung"``) to the #1739
distribution-shift ladder — the writeup's fourth evaluation column, captioned
"random held-out WildChat (conversation-disjoint)". Its contexts are NATURAL
prediction units sampled fresh by ``scripts/issue1739_wcrung_sample.py``: each
row is one real WildChat conversation split at its LAST user turn — the turns
before it are the prefix, that turn is the query — content-hash held out
against the ENTIRE #1092 manifest.

Two structural differences from the pvsynth sibling
(``scripts/issue1739_pvsynth_pod.py``), both consequences of the contexts being
BEHAVIOR-INDEPENDENT (generic chat, no per-behavior instruction pair):

1. **Generation + capture run ONCE, not per behavior.** The same 2,000 contexts
   and the same rollouts serve all three behaviors; only the JUDGE rubric
   differs, and judging is off-GPU. Rollouts are written under the rung-level
   pseudo-behavior ``wildchat`` (:data:`GEN_BEHAVIOR`) and the scoring leg
   re-judges that one pool under each behavior's rubric. Generating three
   times would triple the GPU bill for byte-identical work.
2. **Multi-turn prefixes need a rung-specific renderer.** The shared
   ``generation.render_prompt_parts`` slices the prefix at the FIRST user-turn
   header, which is right for a single-user-turn rung (system persona + query)
   and WRONG for a conversation prefix — it would cut the earlier turns out of
   the prefix while ``capture`` derives ``prefix_end`` from
   ``len(prefix_text)``, silently mis-positioning every prefix-arm read with no
   crash. This leg passes ``render_fn=`` the wcrung renderer
   (``issue1739_wcrung_contexts.render_row_prompt``, the #1092 instruct
   convention), so no existing rung's committed positions move.

Phase sequence (GPU-width discipline: no GPU idles through API-bound work):

    1. contexts: local sample JSON, else stage the shard set from HF
    2. K=5 on-policy rollouts via the production vLLM seam (batched, chunked)
    3. pack the rollout tree into <= 9 MB jsonl shards and upload the TEXT
       FIRST (Upload Policy: text/JSON always, before any reduction — #779);
       packing is mandatory here, not cosmetic: 2,000 x 5 = 10,000 sibling
       JSONs sit exactly at the Hub's 10k-files-per-directory commit cap
    4. reap the vLLM engine, drain the GPU, then batched teacher-forced
       capture (``context_end`` + ``prefix_end`` + ``t1``, all 28 layers)
    5. upload the capture store, then write the completion sentinel LAST

FAN-OUT: the context axis is embarrassingly parallel, so a multi-GPU pod runs
one process per GPU — ``--n-shards <G> --shard-idx <i>`` (round-robin over
contexts, so the 31..7009-token length spread does not hand one shard the long
tail). Each shard MUST get its OWN ``--out-root`` / ``--store-root`` /
``--hf-prefix``: rollout filenames are per-context and would not collide, but
the pack root, the capture store's internal shard numbering, and the sentinel
would. The scoring leg unions the per-shard trees. Width 1 is byte-identical to
the unsharded path.

Capture-recipe parity with the #1092 store is BINDING and inherited by
construction: the same ``generation`` / ``capture`` modules, model, revision,
chat template, layer set, and answer-span (``t1``) semantics — the identical
code path the pvsynth leg used — so wcrung rows join the same representation
space. The judge wave, DV build, and arm scoring are deliberately NOT here.

CONTENT HYGIENE: logs and artifacts carry ids, counts, hashes, and shapes —
never WildChat prompt text or rollout text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_wcrung_pod.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE any torch/vllm import (thread caps + credentials)

# Real-corpus vLLM hang mitigations, ON by default for this rung (gotchas.md
# § "PRE-LAUNCH CHECKLIST": long real-user prompts on multi-GPU RunPod wedge
# vLLM's engine in generate() often enough that the knobs are the default, not
# a post-hang remedy). setdefault, so a launcher may still pin "0". Every
# wcrung cell therefore runs under ONE engine config (the
# one-config-per-comparison rule).
os.environ.setdefault("EPM_VLLM_ENFORCE_EAGER", "1")
os.environ.setdefault("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")

logger = logging.getLogger("issue1739_wcrung_pod")

RUNG = "wildchat_rung"
SPLIT = "eval"
# Contexts are behavior-INDEPENDENT: one rollout pool, three judge rubrics.
GEN_BEHAVIOR = "wildchat"
HF_PREFIX = "issue1739_ctxmap/wildchat_rung"
CONTEXTS_HF_PREFIX = f"{HF_PREFIX}/contexts"
SENTINEL_NAME = "wcrung_capture_done.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung"))
    ap.add_argument(
        "--store-root",
        type=Path,
        default=Path("analysis_tensors/issue_1739/wcrung_store"),
    )
    # This leg is bound to #1739's wildchat rung by construction (it consumes the
    # rung's own sampled contexts and writes the rung's own capture store), so the
    # default names that rung's subtree rather than a reusable destination.
    # UPLOAD_PREFIX_EXEMPT: wildchat-rung-specific leg; --hf-prefix overrides
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument(
        "--rows-json",
        type=Path,
        default=None,
        help="local sampler output (default: <out-root>/contexts/wcrung.json); "
        "absent -> stage the shard set from HF",
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1739/wcrung_stage"),
        help="mirror root for HF-staged context shards",
    )
    ap.add_argument("--k-rollouts", type=int, default=None, help="default: constants.K_ROLLOUTS")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: GEN_MAX_NEW_TOKENS")
    ap.add_argument("--temperature", type=float, default=None, help="default: GEN_TEMPERATURE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--capture-batch-size", type=int, default=None)
    ap.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="SMOKE ONLY: cap the context count (the rung is otherwise all sampled rows)",
    )
    ap.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="fan-out width: partition the contexts across N shards (one process per GPU)",
    )
    ap.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="this process's shard index in [0, n-shards)",
    )
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE ONLY: no Hub writes")
    ap.add_argument("--skip-capture", action="store_true", help="generation only (staging probe)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the REAL branch, then exit 0",
    )
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _upload_dir(local: Path, path_in_repo: str, *, skip: bool) -> str:
    """Bulk folder upload to the data repo (ONE commit — never a per-file loop)."""
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        path_in_repo,
        raise_on_error=True,
    )
    logger.info("[upload] %s -> %s (%s)", local, path_in_repo, url or "no-url")
    return url


def _upload_file(local: Path, path_in_repo: str, *, skip: bool) -> str:
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    return hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        path_in_repo,
        upload_as_file=True,
        raise_on_error=True,
    )


def wcrung_render_fn(tokenizer, row: dict) -> tuple[str, str]:
    """``(prefix_text, prompt_text)`` for a wcrung row — multi-turn safe.

    Delegates to the rung's renderer (the #1092 instruct convention: prefix
    turns rendered AS TURNS, so a conversation prefix keeps them). Passed to
    ``generate_labeling(render_fn=...)`` so the shared single-user-turn default
    — and every rung whose committed ``prefix_end`` positions depend on it —
    stays byte-identical.
    """
    from scripts.issue1739_wcrung_contexts import render_row_prompt

    return render_row_prompt(tokenizer, row.get("prefix_turns") or [], row["query"])


def load_rows(args: argparse.Namespace) -> list[dict]:
    """Sampled context rows: local sampler JSON if present, else staged from HF.

    The 25 MB rows file is deliberately not in git (raw WildChat text), so the
    git-clone-only GPU lane reads the sharded HF copy — the #1773 cross-machine
    seam, in the VM-produced -> clone-lane direction.
    """
    from scripts import issue1739_wcrung_rows_io as rows_io

    local = args.rows_json or (args.out_root / "contexts" / "wcrung.json")
    if local.exists():
        rows = json.loads(local.read_text())["rows"]
        print(f"[phase=wcrung_contexts] local rows={len(rows)} path={local}", flush=True)
        return rows

    rows = rows_io.stage_rows_from_hub(
        hf_prefix=CONTEXTS_HF_PREFIX,
        dest_dir=args.stage_root,
    )
    print(
        f"[phase=wcrung_contexts] staged rows={len(rows)} prefix={CONTEXTS_HF_PREFIX}",
        flush=True,
    )
    return rows


def build_contexts(args: argparse.Namespace) -> list[dict]:
    """Sampled rows -> staged-context rows in the generation schema.

    ``group_key`` is the row's own context id (each conversation is its own
    group), so the ladder's group-level folds stay well-defined: the natural
    units are conversation-disjoint by construction, and no two rows share a
    source conversation.
    """
    rows = load_rows(args)
    if args.max_contexts is not None:
        rows = rows[: args.max_contexts]
    if args.n_shards < 1 or not (0 <= args.shard_idx < args.n_shards):
        raise RuntimeError(f"bad fan-out: --shard-idx {args.shard_idx} not in [0, {args.n_shards})")
    if args.n_shards > 1:
        # ROUND-ROBIN, not contiguous blocks: WildChat prompt lengths span
        # 31..7009 tokens, so a contiguous split would hand one shard the long
        # tail and idle the rest (a work-conserving dispatcher never idles a GPU
        # while an independent cell is pending).
        n_all = len(rows)
        rows = rows[args.shard_idx :: args.n_shards]
        print(
            f"[phase=wcrung_shard] shard {args.shard_idx + 1}/{args.n_shards}: "
            f"{len(rows)}/{n_all} contexts",
            flush=True,
        )
    if not rows:
        raise RuntimeError("no wcrung context rows — sampler output empty or staging failed")

    contexts: list[dict] = []
    for row in rows:
        for key in ("context_id", "query", "group_key"):
            if not row.get(key):
                raise RuntimeError(f"context row missing {key!r} (id={row.get('context_id')!r})")
        contexts.append(
            {
                "context_id": row["context_id"],
                "behavior": GEN_BEHAVIOR,
                # prefix_turns is what wcrung_render_fn renders; prefix_text is
                # the sampler's own render, kept for provenance only.
                "prefix_turns": row.get("prefix_turns") or [],
                "prefix_text": row.get("prefix_text", ""),
                "query": row["query"],
                "split": SPLIT,
                "rung": RUNG,
                "group_key": row["group_key"],
                "source_conv_id": row.get("source_conv_id"),
                "n_prefix_turns": row.get("n_prefix_turns"),
                "single_turn": row.get("single_turn"),
                "query_sha256": row.get("query_sha256"),
            }
        )
    ids = [c["context_id"] for c in contexts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate wcrung context_id in sampled rows")
    return contexts


def reap_generation_engine(drain_timeout_s: int = 180, floor_mib: int = 2048) -> None:
    """Reap the module-cached vLLM engine and DRAIN-WAIT the GPU below a floor.

    vLLM and the HF capture model must NEVER co-reside: the engine reserves
    ``gpu_memory_utilization`` of HBM, so a resident 7B bf16 HF model makes the
    engine init raise (and the reverse starves the capture load). Generation
    runs to completion, then this reap releases the GPU, then the capture model
    loads. The drain verdict reads DEVICE-level ``memory.used`` — never
    compute-apps rows alone, which are pid-visibility-dependent inside a
    container (#825/#1333) and read EMPTY for a foreign holder.
    """
    import subprocess

    from explore_persona_space.experiments.issue_1739 import generation as _gen

    llm = _gen._TOKENIZER_CACHE.pop("_llm", None)
    if llm is None:
        logger.info("[reap] no cached vLLM engine to reap")
        return
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    del llm

    deadline = time.time() + drain_timeout_s
    while True:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        used = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[1].isdigit():
                used.append((int(parts[0]), int(parts[1])))
        worst = max((m for _, m in used), default=0)
        if used and worst <= floor_mib:
            print(f"[phase=wcrung_reap] GPU drained: max_used={worst} MiB", flush=True)
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"vLLM teardown did not drain below {floor_mib} MiB within "
                f"{drain_timeout_s}s (per-GPU used MiB: {used})"
            )
        time.sleep(5)


def generate(args: argparse.Namespace, tokenizer) -> tuple[list[dict], dict]:
    """Build contexts -> K rollouts -> pack + upload rollout TEXT (vLLM only)."""
    from explore_persona_space.experiments.issue_1739 import generation
    from explore_persona_space.experiments.issue_1739.constants import K_ROLLOUTS

    k_rollouts = args.k_rollouts or K_ROLLOUTS
    max_new_tokens = args.max_new_tokens or generation.GEN_MAX_NEW_TOKENS
    temperature = args.temperature if args.temperature is not None else generation.GEN_TEMPERATURE

    contexts = build_contexts(args)
    n_multi = sum(1 for c in contexts if not c.get("single_turn"))
    print(
        f"[phase=wcrung_contexts behavior={GEN_BEHAVIOR}] n_contexts={len(contexts)} "
        f"multi_turn={n_multi} single_turn={len(contexts) - n_multi} k={k_rollouts}",
        flush=True,
    )
    _write_json_atomic(
        args.out_root / "contexts" / "wcrung_gen_contexts.json",
        {
            "rung": RUNG,
            "split": SPLIT,
            "gen_behavior": GEN_BEHAVIOR,
            "n_contexts": len(contexts),
            "n_multi_turn": n_multi,
            # ids + structure only — never query or prefix TEXT.
            "context_ids": [c["context_id"] for c in contexts],
            "git_commit": _git_commit(),
        },
    )

    t0 = time.time()
    gen_manifest = generation.generate_labeling(
        contexts,
        out_root=args.out_root,
        behavior=GEN_BEHAVIOR,
        k_rollouts=k_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=args.seed,
        tokenizer=tokenizer,
        render_fn=wcrung_render_fn,
    )
    print(
        f"[phase=wcrung_generate] kept={gen_manifest['n_kept']} "
        f"generated={gen_manifest['n_generated']} resumed={gen_manifest['n_resumed']} "
        f"truncated={gen_manifest['n_truncated_rollouts']} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    # Rollout TEXT uploads FIRST (durability before any reduction). PACKED:
    # k x n_contexts sibling JSONs would sit at the Hub's 10k-files-per-dir
    # commit cap.
    upload_rollout_text(args)
    return contexts, gen_manifest


def upload_rollout_text(args: argparse.Namespace) -> dict:
    """Pack the rollout tree into <= 9 MB jsonl shards, then upload in ONE commit."""
    from scripts.issue1739_pack import pack_raw_tree

    raw_root = args.out_root / "labeling"
    pack_root = args.out_root / "labeling_packed"
    manifest = pack_raw_tree(raw_root, pack_root)
    n_shards = sum(len(g.get("shards", [])) for g in manifest["groups"].values())
    print(
        f"[phase=wcrung_pack] groups={len(manifest['groups'])} shards={n_shards} root={pack_root}",
        flush=True,
    )
    _upload_dir(
        pack_root,
        f"{args.hf_prefix}/raw_completions_packed",
        skip=args.skip_upload,
    )
    return manifest


def capture(args: argparse.Namespace, gen_manifest: dict, model, tokenizer) -> dict:
    """Batched teacher-forced capture -> upload store (HF model only; no vLLM)."""
    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739.constants import HIDDEN_DIM, N_LAYERS

    rollout_dir = args.out_root / "labeling" / GEN_BEHAVIOR
    rollout_paths = sorted(p for p in rollout_dir.glob("*.json") if not p.name.startswith("_"))
    if not rollout_paths:
        raise RuntimeError(f"no rollout files under {rollout_dir}")
    store_dir = args.store_root / GEN_BEHAVIOR
    t0 = time.time()
    cap_kwargs: dict = {
        "store_dir": store_dir,
        "model": model,
        "tokenizer": tokenizer,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "device": args.device,
        "fingerprint": gen_manifest["fingerprint"],
    }
    if args.capture_batch_size:
        cap_kwargs["batch_size"] = args.capture_batch_size
    cap_manifest = capture_mod.capture_rollout_files(rollout_paths, **cap_kwargs)
    print(
        f"[phase=wcrung_capture] rows={cap_manifest.get('n_rows')} "
        f"shards={cap_manifest.get('n_shards')} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    _upload_dir(
        store_dir,
        f"{args.hf_prefix}/capture_store/{GEN_BEHAVIOR}",
        skip=args.skip_upload,
    )
    return cap_manifest


def _import_check() -> int:
    """Resolve EVERY deferred import on the REAL branch (#1689 Axis 1).

    Lives in its OWN function, NOT inline in ``main()``, and that is
    load-bearing rather than cosmetic. An ``import X`` is a BINDING, so the
    compiler marks X a local of the enclosing function for its WHOLE body —
    including the normal path that never executes the import-check branch. An
    inline block importing the bare name ``capture`` therefore made
    ``main()``'s later call to the MODULE-LEVEL ``def capture(...)`` read an
    unbound local, crashing with ``UnboundLocalError`` at the phase-2 entry
    after generation had already completed. Hoisting the block confines every
    such binding here, where nothing reads a module-level name, so no future
    import added to this list can shadow a module-level symbol in ``main()``.
    Pinned compile-time by test_main_locals_do_not_shadow_module_level_symbols.
    """
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _reap_vllm_engine,
    )
    from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
        capture,
        constants,
        generation,
    )
    from explore_persona_space.experiments.issue_1739.capture import (  # noqa: F401
        capture_rollout_files,
        load_capture_model,
    )
    from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
        HIDDEN_DIM,
        K_ROLLOUTS,
        N_LAYERS,
    )
    from explore_persona_space.experiments.issue_1739.generation import (  # noqa: F401
        GEN_MAX_NEW_TOKENS,
        GEN_TEMPERATURE,
        generate_labeling,
        get_tokenizer,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload,
        stage_hub_prefix,
    )
    from scripts.issue1739_pack import pack_raw_tree  # noqa: F401
    from scripts.issue1739_wcrung_contexts import render_row_prompt  # noqa: F401
    from scripts.issue1739_wcrung_rows_io import (  # noqa: F401
        load_rows as _load_shard_rows,
    )
    from scripts.issue1739_wcrung_rows_io import (  # noqa: F401
        stage_rows_from_hub,
    )

    print("[import-check] OK: all deferred imports resolved", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        return _import_check()

    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739 import generation

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.store_root.mkdir(parents=True, exist_ok=True)

    tokenizer = generation.get_tokenizer()

    # PHASE 1 — generation (vLLM engine loaded once). The HF capture model is
    # NOT loaded yet: co-residency starves the engine's HBM reservation.
    contexts, gen_manifest = generate(args, tokenizer)

    cap_manifest: dict | None = None
    if args.skip_capture:
        print("[phase=wcrung_capture] SKIPPED (--skip-capture)", flush=True)
    else:
        # PHASE 2 — reap the engine, THEN load the capture model, THEN capture.
        reap_generation_engine()
        model = capture_mod.load_capture_model(device=args.device)
        cap_manifest = capture(args, gen_manifest, model, tokenizer)

    sentinel = {
        "rung": RUNG,
        "split": SPLIT,
        "gen_behavior": GEN_BEHAVIOR,
        "judge_behaviors": ["evil", "sycophancy", "hallucination"],
        "hf_prefix": args.hf_prefix,
        "n_shards": args.n_shards,
        "shard_idx": args.shard_idx,
        "n_contexts": gen_manifest["n_contexts"],
        "n_kept": gen_manifest["n_kept"],
        "k_rollouts": gen_manifest["k_rollouts"],
        "n_truncated_rollouts": gen_manifest["n_truncated_rollouts"],
        "gen_fingerprint": gen_manifest["fingerprint"],
        "n_multi_turn_contexts": sum(1 for c in contexts if not c.get("single_turn")),
        "capture_rows": (cap_manifest or {}).get("n_rows"),
        # The binding parity is with the OTHER #1739 rungs (the arms are fit on
        # the #1739 capture stores, all captured through this same
        # load_capture_model default) — not with #1092's own capture call.
        "capture_recipe": "compute bf16, storage fp16, matching all #1739 rung captures",
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sentinel_path = args.out_root / SENTINEL_NAME
    _write_json_atomic(sentinel_path, sentinel)
    # Sentinel LAST: its presence on the Hub means text + tensors already landed.
    _upload_file(sentinel_path, f"{args.hf_prefix}/{SENTINEL_NAME}", skip=args.skip_upload)
    print(f"[phase=done] wcrung capture complete: {sentinel_path}", flush=True)

    # Explicit exit: heavy C-extension teardown can rewrite rc during finalize
    # (gotchas.md PyGILState_Release entry) and abort a set -e dispatcher.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
