"""Issue #1739 tiny-real full-pipeline smoke driver (round C2).

Runs the extract / capture / judge phases through the PRODUCTION CLI
entrypoints (their ``main()`` bodies, byte-identical arguments to the
dispatcher's calls) with EXACTLY the sanctioned fakes patched in at the
external boundaries:

- GPU-scale weights  -> a from-config tiny Qwen2 over the REAL
  Qwen2.5-7B-Instruct vocab (``capture.load_capture_model`` seam);
- vLLM generation    -> a signature-bound stub of
  ``generation._default_vllm_generate``;
- Batch judge API    -> a signature-bound stub of
  ``eval.graded_judge.judge_graded`` returning a REAL ``JudgeResult`` with
  content-drop + transport-loss rows (real drop/split semantics downstream);
- E1 asset REGENERATION -> a signature-bound stub of
  ``scripts.issue779_common.generate_extraction_artifacts`` (the API seam;
  the local-cache leg of the chain stays real);
- real-corpus HF streams -> synthetic fixture streams patched at
  ``corpus_staging._hf_stream`` (the real corpora are probe-verified
  separately by the orchestrator's bounded ingestion probes).

The gates / upload_raw / fits / figures / results phases run through the
REAL dispatcher (``bash scripts/issue1739_dispatch.sh --phase <p>`` with
``EPM_I1739_LIMIT`` + ``EPM_I1739_SMOKE_ROOT``) — the Hub/git stages
dry-run there (sanctioned remote-boundary fake). The #1092 HF STORE read is
REAL: ``--phase realstore`` stages a small slice at the pinned revision
through ``store_io.stage_store_slice`` and runs the production consumer
loaders + whitening/map path at the production 3584 dim.

SMOKE-ROOT DIVERSION (round-2 M4): every OUTPUT this driver writes — staged
corpora, rollout text, capture stores, DV datasets, the tiny u-store/E1
STAND-INS — lands under ``EPM_I1739_SMOKE_ROOT`` (default
``/tmp/i1739-smoke``), matching the dispatcher's own smoke roots, so a
smoke can NEVER write a canonical ``eval_results/`` / ``figures/`` /
``data/issue_1739/`` path (in particular the tiny 64-dim u-store stand-in
must never satisfy the canonical path's loadable predicate — production
would silently consume it). The realstore legs' download caches
(``data/issue_1739/hf_dl/real_store_slice`` / ``u_store_probe``) stay
canonical: they stage the REAL pinned parent inputs, identically in both
modes.

    uv run python scripts/issue1739_smoke_e2e.py --phase extract
    uv run python scripts/issue1739_smoke_e2e.py --phase capture
    ...

CONTENT HYGIENE: every fixture string below is neutral synthetic
placeholder text; nothing is drawn from real corpora or banks.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_smoke_e2e.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
DEFAULT_LIMIT = 25
SMOKE_ROOT = Path(os.environ.get("EPM_I1739_SMOKE_ROOT", "/tmp/i1739-smoke"))
STAGED_ROOT = SMOKE_ROOT / "data/issue_1739/staged"
RAW_ROOT = SMOKE_ROOT / "raw_completions/issue_1739"
STORE_ROOT = SMOKE_ROOT / "data/issue_1739/store"
RESULTS_ROOT = SMOKE_ROOT / "eval_results/issue_1739"
FIGURES_ROOT = SMOKE_ROOT / "figures/issue_1739"
USTORE_STANDIN = SMOKE_ROOT / "data/issue_1739/hf_dl/u_store"
E1_INPUTS = SMOKE_ROOT / "data/issue_1739/inputs"

# ---------------------------------------------------------------------------
# synthetic HF stream fixtures (schemas mirror the real datasets-server rows)
# ---------------------------------------------------------------------------


def _filler(tag: str, i: int, n_words: int = 24) -> str:
    """Per-row UNIQUE synthetic filler: rows must not near-dup each other or
    another dataset's rows (the MinHash train/eval disjointness filter is
    real in this smoke — shared boilerplate across fixtures gets dropped, so
    the filler words are hash-derived and row-unique down to the shingle)."""
    return " ".join(
        hashlib.sha1(f"{tag}-{i}-{j}".encode("utf-8")).hexdigest()[:10] for j in range(n_words)
    )


def _reddit_rows(split: str) -> list[dict]:
    return [
        {
            "id": f"{split}-post{i}",
            "title": f"Synthetic {split} advice title {i}",
            "selftext": f"Synthetic advice body: {_filler(split[:4], i)}.",
            "over_18": "False",
        }
        for i in range(60)
    ]


def _qa_answer(i: int) -> str:
    # ~1/3 of rows share the answer "Paris" (alias-matchable by the stub
    # completions -> exercises the three-way "correct" branch AND a shared
    # group_key); the rest are singleton groups with unmatchable answers.
    return "Paris" if i % 3 == 0 else f"Placeburg{i}"


def synthetic_hf_stream(dataset_id: str, config: str | None, split: str, **kwargs):
    """Signature-bound synthetic twin of ``corpus_staging._hf_stream``."""
    if dataset_id == "HuggingFaceGECLM/REDDIT_submissions":
        return iter(_reddit_rows(split))
    if dataset_id == "TrustAIRLab/in-the-wild-jailbreak-prompts":
        return iter(
            {
                "prompt": f"Synthetic role-play placeholder prefix {i}: {_filler('jb', i)}.",
                "community_id": f"c{i}",
            }
            for i in range(12)
        )
    if dataset_id == "TrustAIRLab/forbidden_question_set":
        return iter(
            {
                "question": f"Synthetic placeholder question {_filler('fq', i, 6)}, please?",
                "q_id": i,
                "content_policy_name": "placeholder",
            }
            for i in range(12)
        )
    if dataset_id == "Anthropic/hh-rlhf":
        return iter(
            {
                "transcript": (
                    f"\n\nHuman: Synthetic placeholder request {i}: {_filler('hh', i)}."
                    "\n\nAssistant: a synthetic placeholder reply."
                )
            }
            for i in range(40)
        )
    if dataset_id == "lmsys/toxic-chat":
        return iter(
            {
                "toxicity": 1,
                "user_input": (
                    f"Synthetic flagged placeholder input {i}: {_filler(split[:2], i)}."
                ),
                "conv_id": f"{split}-conv{i}",
            }
            for i in range(20)
        )
    if dataset_id == "mandarjoshi/trivia_qa":
        return iter(
            {
                "question": f"Synthetic trivia question {_filler('tq', i, 6)}: which city?",
                "question_id": f"tq{i}",
                "answer": {"value": _qa_answer(i), "aliases": [_qa_answer(i)]},
            }
            for i in range(40)
        )
    if dataset_id == "google-research-datasets/nq_open":
        return iter(
            {
                "question": f"synthetic open question {_filler('nq', i, 6)}: which city?",
                "answer": [_qa_answer(i)],
            }
            for i in range(40)
        )
    if dataset_id == "basicv8vc/SimpleQA":
        return iter(
            {
                "problem": f"Synthetic simple question {_filler('sq', i, 6)}: which city?",
                "answer": _qa_answer(i),
            }
            for i in range(30)
        )
    raise KeyError(f"no synthetic fixture for dataset {dataset_id!r}")


# ---------------------------------------------------------------------------
# boundary stubs (signature-bound)
# ---------------------------------------------------------------------------

_COMPLETION_TEMPLATES = (
    "The answer is Paris. This synthetic placeholder reply agrees confidently.",
    "I do not know the answer to that question; this synthetic reply declines to guess.",
    "The answer is definitely Placeville. This synthetic reply asserts an unsupported claim.",
    "A hedged synthetic reply: it could be one of several places, hard to say for sure.",
)


def stub_vllm_generate(
    prompts: list[str],
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    seeds: list[int],
) -> list[list[dict]]:
    """Signature twin of ``generation._default_vllm_generate`` (no GPU)."""
    out: list[list[dict]] = []
    for prompt, seed in zip(prompts, seeds, strict=True):
        h = int(hashlib.sha1(prompt.encode("utf-8")).hexdigest(), 16)
        out.append(
            [
                {
                    "text": _COMPLETION_TEMPLATES[(h + k + seed) % len(_COMPLETION_TEMPLATES)],
                    "finish_reason": "stop",
                }
                for k in range(n)
            ]
        )
    return out


def stub_generate_extraction_artifacts(trait: str, *, force: bool = False) -> dict:
    """Signature twin of ``issue779_common.generate_extraction_artifacts``
    (the Sonnet API seam of the E1 asset chain)."""
    return {
        "instruction": [
            {
                "pos": f"You are a persona that strongly exhibits synthetic trait "
                f"{trait} (pair {p}).",
                "neg": f"You are a persona that never exhibits synthetic trait {trait} (pair {p}).",
            }
            for p in range(5)
        ],
        "extraction_questions": [f"Synthetic neutral extraction question {q}?" for q in range(20)],
        "eval_prompt": (
            "Rate from 0 to 100 how strongly the answer expresses the synthetic "
            "trait. Reason briefly, then give the integer.\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    }


def stub_judge_graded(
    items: list[tuple[str, str, str]],
    eval_prompt: str,
    *,
    n_draws: int,
    cache_dir: Path,
    save_raw: Path,
    judge_model: str = "stub",
    temperature: float = 1.0,
    max_tokens: int = 64,
    dry_run: bool = False,
    threshold_base: int | None = None,
):
    """Signature twin of ``eval.graded_judge.judge_graded`` — stubbed
    TRANSPORT only: returns a REAL ``JudgeResult`` carrying content drops
    (one partial-draw item + one all-draws-dropped item) and a transport
    loss, so the drop/split semantics downstream run for real."""
    from explore_persona_space.eval.graded_judge import JudgeResult

    scores: dict[str, float | None] = {}
    per_scores: dict[str, list[float]] = {}
    per_counts: dict[str, int] = {}
    per_transport: dict[str, int] = {}
    n_dropped = 0
    n_transport = 0
    for i, (item_id, _q, answer) in enumerate(items):
        if i == 1 and len(items) > 3:
            scores[item_id] = None  # all draws content-dropped
            per_scores[item_id] = []
            per_counts[item_id] = 0
            n_dropped += n_draws
            continue
        base = float(int(hashlib.sha1((item_id + answer).encode()).hexdigest(), 16) % 101)
        draws = [min(100.0, max(0.0, base + d)) for d in range(n_draws)]
        if i == 0 and n_draws > 1:
            draws = draws[:-1]  # one content-dropped draw
            n_dropped += 1
        if i == 2 and len(items) > 3:
            per_transport[item_id] = 1
            n_transport += 1
        scores[item_id] = sum(draws) / len(draws)
        per_scores[item_id] = draws
        per_counts[item_id] = len(draws)
    save_raw = Path(save_raw)
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    save_raw.write_text(json.dumps({"stub_transport": True, "n_items": len(items)}))
    return JudgeResult(
        scores=scores,
        n_total_draws=n_draws * len(items),
        n_dropped_draws=n_dropped,
        per_item_scores=per_scores,
        per_item_draw_counts=per_counts,
        n_transport_lost_draws=n_transport,
        per_item_transport_losses=per_transport,
    )


_TINY_MODEL_CACHE: dict = {}


def tiny_real_model_loader(device: str = "cuda", dtype: str = "bfloat16"):
    """Signature twin of ``capture.load_capture_model``: from-config tiny
    Qwen2 (4 layers, hidden 64) over the REAL Qwen2.5-7B-Instruct vocab."""
    if "model" in _TINY_MODEL_CACHE:
        return _TINY_MODEL_CACHE["model"]
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1739)
    config = Qwen2Config(
        vocab_size=152064,  # real Qwen2.5-7B-Instruct config vocab (covers all ids)
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).to("cpu")
    model.eval()
    _TINY_MODEL_CACHE["model"] = model
    return model


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


def _run_cli_main(module_name: str, argv: list[str]) -> None:
    import importlib

    cli = importlib.import_module(module_name)
    with mock.patch.object(sys, "argv", [module_name.rsplit(".", 1)[-1] + ".py", *argv]):
        rc = cli.main()
    assert rc == 0, (module_name, rc, argv)


def phase_extract(behaviors: tuple[str, ...], limit: int) -> None:
    """Mirror of the dispatcher extract phase: staging + labeling gen + E1 gen."""
    from explore_persona_space.experiments.issue_1739 import corpus_staging, generation
    from explore_persona_space.experiments.issue_1739.corpus_staging import stage_corpus

    import scripts.issue779_common as issue779_common

    with mock.patch.object(corpus_staging, "_hf_stream", synthetic_hf_stream):
        for b in behaviors:
            print(f"[smoke-extract] staging behavior={b}", flush=True)
            stage_corpus(b, "train", limit, 0, out_dir=STAGED_ROOT / b)
            stage_corpus(b, "eval", limit, 0, out_dir=STAGED_ROOT / b)
    with (
        mock.patch.object(generation, "_default_vllm_generate", stub_vllm_generate),
        mock.patch.object(
            issue779_common, "generate_extraction_artifacts", stub_generate_extraction_artifacts
        ),
    ):
        for b in behaviors:
            print(f"[smoke-extract] labeling generation behavior={b}", flush=True)
            ctx_glob = sorted(glob.glob(f"{STAGED_ROOT}/{b}/{b}_*_*.contexts.jsonl"))
            assert ctx_glob, b
            _run_cli_main(
                "scripts.issue1739_generate",
                [
                    "--mode",
                    "labeling",
                    "--behavior",
                    b,
                    "--contexts-jsonl",
                    *ctx_glob,
                    "--out-root",
                    str(RAW_ROOT),
                    "--max-contexts",
                    str(limit),
                ],
            )
            print(f"[smoke-extract] E1 extraction generation behavior={b}", flush=True)
            _run_cli_main(
                "scripts.issue1739_generate",
                [
                    "--mode",
                    "extraction",
                    "--behavior",
                    b,
                    "--out-root",
                    str(RAW_ROOT),
                    "--inputs-dir",
                    str(E1_INPUTS),
                    "--n-rollouts",
                    "2",
                ],
            )
    print("[smoke-extract] done", flush=True)


def phase_capture(behaviors: tuple[str, ...], limit: int) -> None:
    """Mirror of the dispatcher capture phase (+ the tiny U-store stand-in)."""
    from explore_persona_space.experiments.issue_1739 import capture as capture_mod

    with mock.patch.object(capture_mod, "load_capture_model", tiny_real_model_loader):
        for b in behaviors:
            print(f"[smoke-capture] capture behavior={b}", flush=True)
            _run_cli_main(
                "scripts.issue1739_capture",
                [
                    "--rollout-dir",
                    f"{RAW_ROOT}/labeling/{b}",
                    "--store-dir",
                    f"{STORE_ROOT}/{b}_labeling",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ],
            )
            print(f"[smoke-capture] E1 extraction capture behavior={b}", flush=True)
            _run_cli_main(
                "scripts.issue1739_capture",
                [
                    "--rollout-dir",
                    f"{RAW_ROOT}/extraction/{b}",
                    "--store-dir",
                    f"{STORE_ROOT}/{b}_extraction",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ],
            )
        # Tiny dim-matched U-store stand-in for the staged #1092 slice (the
        # REAL staged-slice read runs in --phase realstore at 3584 dim; a
        # 3584-dim U pool cannot join a 64-dim tiny labeled store).
        print("[smoke-capture] tiny U-store stand-in", flush=True)
        _run_cli_main(
            "scripts.issue1739_capture",
            [
                "--rollout-dir",
                f"{RAW_ROOT}/labeling/sycophancy",
                "--store-dir",
                str(USTORE_STANDIN),
                "--limit",
                str(10 * limit),
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ],
        )
    print("[smoke-capture] done", flush=True)


def phase_judge(behaviors: tuple[str, ...], limit: int) -> None:
    """Mirror of the dispatcher judge phase (transport stubbed at judge_graded)."""
    import explore_persona_space.eval.graded_judge as graded_judge_mod

    with mock.patch.object(graded_judge_mod, "judge_graded", stub_judge_graded):
        for b in behaviors:
            print(f"[smoke-judge] judge behavior={b}", flush=True)
            _run_cli_main(
                "scripts.issue1739_judge",
                [
                    "--behavior",
                    b,
                    "--rollout-dir",
                    f"{RAW_ROOT}/labeling/{b}",
                    "--out-dir",
                    f"{RESULTS_ROOT}/judge/{b}",
                    "--dv-out-root",
                    str(RESULTS_ROOT),
                ],
            )
    print("[smoke-judge] done", flush=True)


def phase_gates12(behaviors: tuple[str, ...], limit: int) -> None:
    """Gates 1-2 (production gate functions) over the smoke DV datasets."""
    from explore_persona_space.experiments.issue_1739 import gates

    out_dir = RESULTS_ROOT / "gate12_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in behaviors:
        dv = json.loads((RESULTS_ROOT / f"dv_dataset/{b}/labeling.json").read_text())
        g1 = gates.gate1_yield_report(dv["rows"], behavior=b, n_pilot=limit)
        g2 = gates.gate2_spread_floor(dv["rows"], behavior=b)
        (out_dir / f"{b}.json").write_text(json.dumps({"gate1": g1, "gate2": g2}, indent=2))
        print(
            f"[smoke-gates12] {b}: gate1={g1['verdict']} keep_rate={g1['keep_rate']:.2f} "
            f"gate2={g2['verdict']} sd={g2.get('inter_context_sd')}",
            flush=True,
        )
    print("[smoke-gates12] done", flush=True)


def phase_realstore() -> None:
    """REAL #1092 store read through the PRODUCTION layout-mapping adapter:

    (a) dynamics_instruct leg — canonical kind names (context_end/t1) resolve
        via the realized aliases (context_k/answer_k_t1) + per-kind row_index
        stems inside ``store_io.load_summaries`` (the composite consumer),
        then whitening + linear map at the production 3584 dim;
    (b) U-store cell leg — the PRODUCTION fits mapping:
        ``store_io.stage_u_store`` (cell_inst_own shards flattened + corpus
        manifest.jsonl as row metadata, 1-shard probe slice) ->
        ``load_summaries`` -> ``fit_pool_mask``;
    (c) the REAL r_B bank through its consumer loader.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io
    from explore_persona_space.experiments.issue_1739.constants import (
        HIDDEN_DIM,
        N_LAYERS,
        RB_N_TRAITS,
        SUMMARY_KINDS,
    )

    local = Path("data/issue_1739/hf_dl/real_store_slice")
    t0 = time.time()
    # (a) dynamics leg: request CANONICAL kinds; the adapter maps them to the
    # realized names + per-kind row_index stems (round-C2 findings, now wired).
    kinds = ("context_end", "t1")
    staged = store_io.stage_store_slice(kinds, (0,), 64, local, cell="dynamics_instruct")
    print(f"[smoke-realstore] staged {len(staged)} files in {time.time() - t0:.0f}s", flush=True)
    arrays, meta = store_io.load_summaries(local, kinds, (0,), cell="dynamics_instruct", n_rows=64)
    row_keys = sorted(meta[0])
    print(
        f"[smoke-realstore] dynamics rows: n={len(meta)} row_index keys: {row_keys}",
        flush=True,
    )
    mask = store_io.fit_pool_mask(meta)
    rows = np.flatnonzero(mask)
    x_u = np.stack([arrays[("context_end", 0)][rows].astype(np.float64)])
    y_u = np.stack([arrays[("t1", 0)][rows].astype(np.float64)])
    wh = fits.fit_whitening(x_u, device="cpu")
    mapfit = fits.fit_linear_map(
        fits.apply_whitening(x_u, wh), fits.apply_whitening(y_u, wh), device="cpu"
    )
    print(
        f"[smoke-realstore] real-dim whitening+map OK: n={len(rows)} d={HIDDEN_DIM} "
        f"diag_keys={sorted(mapfit.diagnostics)[:6]}",
        flush=True,
    )
    # (b) U-store cell leg — the exact mapping production fits consumes.
    u_local = Path("data/issue_1739/hf_dl/u_store_probe")
    u_root = store_io.stage_u_store(u_local, SUMMARY_KINDS, (0,), max_shards_per_kind=1)
    u_arrays, u_meta = store_io.load_summaries(u_root, SUMMARY_KINDS, (0,), n_rows=16)
    u_mask = store_io.fit_pool_mask(u_meta)
    u_shapes = {f"{k}_L{ly:02d}": list(a.shape) for (k, ly), a in u_arrays.items()}
    n_manifest = len(store_io._iter_jsonl(u_local / "manifest.jsonl"))
    print(
        f"[smoke-realstore] u_store cell leg OK: shapes={u_shapes} "
        f"manifest_rows={n_manifest} fit_kept={int(u_mask.sum())}/16 "
        f"meta_keys={sorted(u_meta[0])}",
        flush=True,
    )
    # (c) r_B bank through its consumer loader.
    bank, names = store_io.load_rb_bank()
    assert bank.shape == (N_LAYERS, RB_N_TRAITS, HIDDEN_DIM), bank.shape
    print(f"[smoke-realstore] r_B bank OK: shape={bank.shape} traits={names}", flush=True)
    print(
        "REALSTORE SMOKE: PASS shapes="
        f"dynamics(n={len(meta)},d={HIDDEN_DIM}),u_store={u_shapes},"
        f"manifest_rows={n_manifest},rb={list(bank.shape)}",
        flush=True,
    )
    print("[smoke-realstore] done", flush=True)


def phase_figcheck() -> None:
    """Open one rendered PNG + its sidecar; assert non-empty axes/points."""
    import numpy as np
    from PIL import Image

    pngs = sorted(FIGURES_ROOT.rglob("*.png"))
    assert pngs, "no figure PNGs rendered"
    checked = 0
    for png in pngs:
        arr = np.asarray(Image.open(png).convert("L"), dtype=np.float64)
        assert arr.size > 10_000 and arr.std() > 1.0, (png, arr.shape, float(arr.std()))
        meta_path = png.with_name(png.name.replace(".png", ".meta.json"))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"[smoke-figcheck] {png.name}: std={arr.std():.1f} meta_keys={sorted(meta)[:5]}")
        checked += 1
    print(f"[smoke-figcheck] {checked} PNGs non-blank; done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase",
        required=True,
        choices=["extract", "capture", "judge", "gates12", "realstore", "figcheck"],
    )
    parser.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    behaviors = tuple(args.behaviors)
    t0 = time.time()
    if args.phase == "extract":
        phase_extract(behaviors, args.limit)
    elif args.phase == "capture":
        phase_capture(behaviors, args.limit)
    elif args.phase == "judge":
        phase_judge(behaviors, args.limit)
    elif args.phase == "gates12":
        phase_gates12(behaviors, args.limit)
    elif args.phase == "realstore":
        phase_realstore()
    else:
        phase_figcheck()
    print(f"[smoke] phase={args.phase} rc=0 elapsed={time.time() - t0:.0f}s", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
