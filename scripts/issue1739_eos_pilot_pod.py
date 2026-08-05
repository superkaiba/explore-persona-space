"""Issue #1739 evil-ood-spread: pod-side pilot ROLLOUT GENERATION (item A, plan v16 §4.4).

One corpus per pod (A1a MHJ / A1b tom-gibbs / A2 PAIR): read the VM-prepped
context JSONL (``scripts/issue1739_evil_rung_gen.py`` output, committed on the
issue branch), generate K rollouts per context with vLLM, mirror the rollout
JSONs into the layout ``scripts/issue1739_pilot_judge.py`` reads, upload the
rollout TEXT to the HF data repo, then write the done sentinel LAST.

Why this driver exists: ``issue1739_evil_rung_gen.py`` is corpus PREP only (it
writes contexts and returns) and ``issue1739_pilot_judge.py`` consumes rollouts
at ``<root>/<rung>/rollouts/<split>/*.json``. Nothing bridged the two.

Reuse (CLAUDE.md "reuse existing experiment code"):
  * ``experiments.issue_1739.generation.generate_labeling`` — batched vLLM
    generation, per-context resume, prompt-budget length gate (#952), atomic
    per-rollout writes, per-unit progress lines.
  * ``scripts.issue1739_wcrung_contexts.render_row_prompt`` — the #1092 INSTRUCT
    multi-turn renderer (prefix turns rendered AS TURNS, so a conversation
    prefix keeps them; ``generation.render_prompt_parts`` would slice inside it).
  * ``orchestrate.hub`` — one bulk folder commit, never a per-file loop (#664).

Turn semantics (both multi-turn corpora publish the ATTACKER's successive
messages and NO target replies): every prior turn is rendered as a ``user``
turn and the LAST turn is the query. No assistant turn is fabricated — putting
attack text in the assistant role is a different (prefill-shaped) stimulus.
A faithful interactive replay (generate each intermediate reply) is out of
pilot scope and is stated as a deviation in the round report.

Usage (pod, production):

    uv run python scripts/issue1739_eos_pilot_pod.py --corpus mhj

Smoke (CPU-runnable pre-generation portion; no GPU, no Hub writes):

    uv run python scripts/issue1739_eos_pilot_pod.py --corpus mhj --prep-only \\
        --contexts-jsonl <path> --out-root /tmp/eos_smoke --skip-upload
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials + shared-VM thread caps bind BEFORE any heavy import (#847).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

logger = logging.getLogger(__name__)


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on ``sys.path`` (script mode puts only scripts/ there, #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_eos_pilot_pod.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIOR = "evil"
SPLIT = "pilot"

# corpus slug -> (rung DIRECTORY name the pilot judge reads, pod name suffix)
CORPUS_RUNG = {"mhj": "mhj", "tomgibbs": "tom-gibbs", "pair": "pair"}
CORPUS_SUFFIX = {"mhj": "a1apilot", "tomgibbs": "a1bpilot", "pair": "a2pilot"}

# plan v16 §4.7 raw-rollout destination
HF_ROOT_PREFIX = "issue1739_ctxmap/raw_completions/evil_ood_spread"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="#1739 evil-ood-spread pilot rollout generation (one corpus per pod)."
    )
    ap.add_argument("--corpus", choices=sorted(CORPUS_RUNG), required=True)
    ap.add_argument("--split", default=SPLIT, help="rollout split id (default pilot)")
    ap.add_argument(
        "--contexts-jsonl",
        default=None,
        help="context JSONL (default: <out-root>/contexts/evil_rung_<corpus>.jsonl)",
    )
    ap.add_argument(
        "--out-root",
        default="eval_results/issue_1739/evil_ood_spread",
        help="round out-root (rollouts land under <out-root>/<rung>/)",
    )
    ap.add_argument("--k-rollouts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-contexts", type=int, default=None, help="smoke slice cap")
    ap.add_argument("--hf-prefix", default=None, help=f"default {HF_ROOT_PREFIX}/<rung>")
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE ONLY: no Hub writes")
    ap.add_argument(
        "--prep-only",
        action="store_true",
        help="SMOKE: build + render contexts (CPU), skip generation/upload/sentinel",
    )
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports; exit")
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        return out.stdout.strip() or "unavailable"
    except OSError:
        return "unavailable"


def eos_render_fn(tokenizer, row: dict) -> tuple[str, str]:
    """``(prefix_text, prompt_text)`` for a possibly multi-turn attack context."""
    from scripts.issue1739_wcrung_contexts import render_row_prompt

    return render_row_prompt(tokenizer, row.get("prefix_turns") or [], row["query"])


def stage_contexts(path: Path, *, corpus: str, hf_prefix: str, skip: bool) -> Path:
    """Stage the corpus-prep JSONL from the HF data repo when absent locally.

    The context sets are free-text ATTACK corpora (~0.25-0.8 MB each), so they
    live on the HF data repo rather than in git (upload-policy.md § large
    free-text JSONs). ``stage_hub_file`` is the canonical retried + atomic +
    fail-loud staging helper (#1402).
    """
    if path.exists():
        logger.info("[stage] contexts present locally: %s", path)
        return path
    if skip:
        raise RuntimeError(f"context JSONL missing and --skip-upload set: {path}")
    from explore_persona_space.orchestrate import hub

    path_in_repo = f"{hf_prefix}/evil_rung_{corpus}.jsonl"
    print(f"[phase=eos_stage] staging {path_in_repo} -> {path}", flush=True)
    return hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, path_in_repo, path, repo_type="dataset")


def read_context_records(path: Path) -> list[dict]:
    """Read the corpus-prep JSONL (text-mode iteration, never splitlines(), #825)."""
    if not path.exists():
        raise RuntimeError(
            f"context JSONL missing: {path} — run scripts/issue1739_evil_rung_gen.py on the "
            "VM and upload its output to the HF data repo (see stage_contexts)"
        )
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"context JSONL empty: {path}")
    return records


def build_contexts(records: list[dict], *, corpus: str, split: str, cap: int | None) -> list[dict]:
    """Corpus-prep records -> ``generate_labeling`` context rows.

    ``turns`` (attacker messages, oldest first) becomes ``prefix_turns`` (all
    but the last, each ``user``) + ``query`` (the last). ``group_key`` is the
    row's own id: each attack conversation is its own group, so any downstream
    group-level fold stays well defined.
    """
    if cap is not None:
        records = records[:cap]
    contexts: list[dict] = []
    for rec in records:
        cid = rec.get("context_id")
        turns = [t for t in (rec.get("turns") or []) if str(t).strip()]
        if not turns:
            # Legacy/prep fallback: a pre-joined transcript with no structure.
            ctx = str(rec.get("context") or "").strip()
            if not ctx:
                raise RuntimeError(f"context row {cid!r} has neither 'turns' nor 'context'")
            turns = [ctx]
        if not cid:
            raise RuntimeError("context row missing context_id")
        contexts.append(
            {
                "context_id": cid,
                "behavior": BEHAVIOR,
                "prefix_turns": [{"role": "user", "content": str(t)} for t in turns[:-1]],
                "query": str(turns[-1]),
                "split": split,
                "rung": rec.get("rung") or f"evil_{corpus}",
                "group_key": cid,
                "n_turns": len(turns),
                "single_turn": len(turns) == 1,
                "source_meta": rec.get("meta") or {},
            }
        )
    ids = [c["context_id"] for c in contexts]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate context_id in {corpus} context set")
    return contexts


def mirror_to_judge_layout(
    *, gen_root: Path, judge_dir: Path, contexts: list[dict], k_rollouts: int
) -> int:
    """Copy per-rollout JSONs into ``<rung>/rollouts/<split>/`` for the pilot judge.

    ``generate_labeling`` owns its own path convention (and its resume
    fingerprint keys on it), so the judge-facing layout is a MIRROR rather than
    a redirect: both trees persist, and a re-run is idempotent.
    """
    from explore_persona_space.experiments.issue_1739.generation import labeling_rollout_path

    judge_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    missing: list[str] = []
    for ctx in contexts:
        for k in range(k_rollouts):
            src = labeling_rollout_path(gen_root, BEHAVIOR, ctx["context_id"], k)
            if not src.exists():
                missing.append(f"{ctx['context_id']}_seed{k}")
                continue
            shutil.copyfile(src, judge_dir / src.name)
            n += 1
    if missing:
        # Budget-dropped contexts (#952 length gate) legitimately have no
        # rollouts. The CALLER raises on any shortfall against
        # n_kept x k_rollouts (which already excludes budget drops), so this
        # is a diagnostic line, not the invariant.
        logger.info("[mirror] %d rollout files absent (budget-dropped contexts)", len(missing))
    return n


def _upload_dir(local: Path, path_in_repo: str, *, skip: bool) -> str:
    """ONE bulk folder commit to the data repo (never a per-file loop, #664)."""
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        local,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    logger.info("[upload] %s -> %s (%s)", local, path_in_repo, url or "no-url")
    return url or ""


def _upload_file(local: Path, path_in_repo: str, *, skip: bool) -> str:
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    return (
        hub._upload(
            local,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            upload_as_file=True,
        )
        or ""
    )


def _import_check() -> int:
    """Resolve EVERY deferred import on the real code path (#606/#1689)."""
    from explore_persona_space.experiments.issue_1739.generation import (  # noqa: F401
        GEN_MAX_NEW_TOKENS,
        GEN_TEMPERATURE,
        MODEL_NAME,
        generate_labeling,
        get_tokenizer,
        labeling_rollout_path,
    )
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload,
        stage_hub_file,
    )
    from scripts.issue1739_wcrung_contexts import render_row_prompt  # noqa: F401

    print("[import-check] OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if args.import_check:
        return _import_check()

    from explore_persona_space.experiments.issue_1739 import generation

    rung = CORPUS_RUNG[args.corpus]
    out_root = Path(args.out_root)
    ctx_path = (
        Path(args.contexts_jsonl)
        if args.contexts_jsonl
        else out_root / "contexts" / f"evil_rung_{args.corpus}.jsonl"
    )
    rung_root = out_root / rung
    gen_root = rung_root / "gen"
    judge_dir = rung_root / "rollouts" / args.split
    hf_prefix = args.hf_prefix or f"{HF_ROOT_PREFIX}/{rung}"
    temperature = args.temperature if args.temperature is not None else generation.GEN_TEMPERATURE

    ctx_path = stage_contexts(
        ctx_path,
        corpus=args.corpus,
        hf_prefix=f"{HF_ROOT_PREFIX}/contexts",
        skip=args.skip_upload,
    )
    records = read_context_records(ctx_path)
    contexts = build_contexts(records, corpus=args.corpus, split=args.split, cap=args.max_contexts)
    n_multi = sum(1 for c in contexts if not c["single_turn"])
    print(
        f"[phase=eos_contexts corpus={args.corpus} rung={rung}] n_contexts={len(contexts)} "
        f"multi_turn={n_multi} single_turn={len(contexts) - n_multi} k={args.k_rollouts} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    _write_json_atomic(
        rung_root / "contexts_manifest.json",
        {
            "corpus": args.corpus,
            "rung": rung,
            "split": args.split,
            "n_contexts": len(contexts),
            "n_multi_turn": n_multi,
            "turns_min": min(c["n_turns"] for c in contexts),
            "turns_max": max(c["n_turns"] for c in contexts),
            # ids + structure only — never query or prefix TEXT.
            "context_ids": [c["context_id"] for c in contexts],
            "contexts_jsonl": str(ctx_path),
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

    tokenizer = generation.get_tokenizer()
    # Render every context on CPU BEFORE any GPU work: a render/prefix defect
    # then fails in seconds instead of after the engine loads.
    for ctx in contexts:
        prefix_text, prompt_text = eos_render_fn(tokenizer, ctx)
        if not prompt_text.startswith(prefix_text):
            raise RuntimeError(f"prefix is not a prefix of the prompt for {ctx['context_id']!r}")
    print(f"[phase=eos_render] rendered {len(contexts)} prompts (prefix invariant OK)", flush=True)

    if args.prep_only:
        # NOT [phase=done]: a mid-pipeline reserved-token emission reads as a
        # false status=done to poll_pipeline.py (#545/#920).
        print("[eos-prep-only] contexts built + rendered; no generation", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    t0 = time.time()
    gen_manifest = generation.generate_labeling(
        contexts,
        out_root=gen_root,
        behavior=BEHAVIOR,
        k_rollouts=args.k_rollouts,
        temperature=temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        tokenizer=tokenizer,
        render_fn=eos_render_fn,
    )
    print(
        f"[phase=eos_generate corpus={args.corpus}] kept={gen_manifest['n_kept']} "
        f"generated={gen_manifest['n_generated']} resumed={gen_manifest['n_resumed']} "
        f"truncated={gen_manifest['n_truncated_rollouts']} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    n_mirrored = mirror_to_judge_layout(
        gen_root=gen_root, judge_dir=judge_dir, contexts=contexts, k_rollouts=args.k_rollouts
    )
    print(f"[phase=eos_mirror] {n_mirrored} rollout JSONs -> {judge_dir}", flush=True)
    if n_mirrored == 0:
        raise RuntimeError("no rollout JSONs mirrored — generation produced nothing")
    # Exact-count invariant: generate_labeling KEPT contexts each own K rollout
    # files, so a short mirror is a generation gap and must be loud (the
    # budget-dropped contexts are already excluded from n_kept).
    n_expected = int(gen_manifest["n_kept"]) * int(args.k_rollouts)
    if n_mirrored != n_expected:
        raise RuntimeError(
            f"mirror gap: {n_mirrored} rollout JSONs mirrored, expected "
            f"{n_expected} (n_kept={gen_manifest['n_kept']} x k={args.k_rollouts})"
        )

    # Rollout TEXT uploads BEFORE the sentinel (durability first).
    # UPLOAD_PREFIX_EXEMPT: single-issue pilot driver; hf_prefix is issue-1739-scoped by construction (CORPUS_RUNG/CORPUS_SUFFIX/sentinel names), and --hf-prefix overrides it
    _upload_dir(judge_dir, f"{hf_prefix}/rollouts/{args.split}", skip=args.skip_upload)
    # UPLOAD_PREFIX_EXEMPT: single-issue pilot driver; hf_prefix is issue-1739-scoped by construction, and --hf-prefix overrides it
    _upload_file(
        rung_root / "contexts_manifest.json",
        f"{hf_prefix}/contexts_manifest.json",
        skip=args.skip_upload,
    )

    sentinel = {
        "issue": 1739,
        "phase": "done",
        "status": "ok",
        "corpus": args.corpus,
        "rung": rung,
        "split": args.split,
        "behavior": BEHAVIOR,
        "n_contexts": gen_manifest["n_contexts"],
        "n_kept": gen_manifest["n_kept"],
        "k_rollouts": gen_manifest["k_rollouts"],
        "n_truncated_rollouts": gen_manifest["n_truncated_rollouts"],
        "n_rollout_files": n_mirrored,
        "gen_fingerprint": gen_manifest["fingerprint"],
        "max_new_tokens": args.max_new_tokens,
        "temperature": temperature,
        "hf_prefix": hf_prefix,
        "judge_rollout_dir": str(judge_dir),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    slug = CORPUS_SUFFIX[args.corpus]
    for sentinel_path in (
        Path(f"/workspace/logs/issue-1739-{slug}-done.json"),
        rung_root / f"eos_pilot_{args.corpus}_done.json",
    ):
        try:
            _write_json_atomic(sentinel_path, sentinel)
        except OSError as exc:  # /workspace absent off-pod
            logger.warning("[sentinel] could not write %s: %s", sentinel_path, exc)
    # UPLOAD_PREFIX_EXEMPT: single-issue pilot driver; hf_prefix is issue-1739-scoped by construction, and --hf-prefix overrides it
    _upload_file(
        rung_root / f"eos_pilot_{args.corpus}_done.json",
        f"{hf_prefix}/eos_pilot_{args.corpus}_done.json",
        skip=args.skip_upload,
    )
    # The RESERVED [phase=done] token belongs to the launcher's single terminal
    # line (issue1739_eos_pilot_launch.sh emits it on rc=0); a mid-pipeline
    # emission from this child would read as a false status=done to
    # poll_pipeline.py (#545/#920).
    print(
        f"[eos-pilot-complete] {args.corpus}: {n_mirrored} rollout JSONs, sentinel written",
        flush=True,
    )

    # Explicit exit: heavy C-extension teardown can rewrite rc during finalize
    # (gotchas.md PyGILState_Release entry) and abort a set -e dispatcher.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    # sys.exit(main()) — a future non-zero return must not exit 0 silently.
    sys.exit(main())
