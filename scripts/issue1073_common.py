"""Shared constants + helpers for issue #1073 (decoding-regime robustness of h).

Everything the #1073 phase entrypoints (``issue1073_gen.py`` P0-P2,
``issue1073_capture.py`` P3, ``issue1073_fits.py`` P4, ``issue1073_figures.py``
P5) share:

- Reused-input HF fetch at the PINNED revision (plan §10: per-file
  ``hf_hub_download`` — never ``snapshot_download`` on the ~1M-file data repo;
  pass_a staged via server-side-scoped ``list_repo_tree``).
- Decode-arm sampling params (plan §11: arm (c)/(a) = the verbatim
  ``issue779_collect.py`` pass-B recipe; arm (b) = greedy).
- String-normalized exact-duplicate clustering + DUPLICATE-CLUSTERED fold
  assignment (plan §12 assumption 11 statistics Must-Fix; the seed-0
  reproduction-gate pass keeps raw pointwise folds for byte-parity).
- <9 MB text-shard writer + verified HF ``upload_folder`` (upload-policy:
  text rides the non-LFS path, never gzip).
- Tiny-real smoke fixtures: a from-config 2-layer same-arch model over the
  REAL tokenizer vocab, a fixture pass-B bundle built through the SAME
  imported parent capture functions, schema-conformant pass_a cells + r_b.
- The pod-side sentinel writer (``issue779_common.write_sentinel`` with
  ``task_id=1073``) + ``[phase=...]`` breadcrumbs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("issue1073")

TASK_ID = 1073
HF_DATA_REPO = C.HF_DATA_REPO
HF_PREFIX = "issue1073_decode_regime"

# Plan §10: every reused #779 input is fetched at this pinned revision.
PINNED_REVISION = "037fcbb210bc52c459959b0746cc268fe08bae96"
BUNDLE_PATH_IN_REPO = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
PASS_A_PREFIX = "issue779_monitoring/analysis_tensors/pass_a"
RB_PREFIX = "issue779_monitoring/r_b"
STEP0_PATH_IN_REPO = "issue779_monitoring/analysis_tensors/step0/step0_oracle.json"

# Decode params (plan §10 "Decode params" — arm (c)/(a): issue779_collect.py:558/:314
# verbatim; arm (b): theory plan C11 greedy with max_tokens matched to (a)/(c)).
SP_STOCH1 = {"n": 1, "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": 42}
SP_STOCH10 = {"n": 10, "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": 42}
SP_GREEDY = {"n": 1, "temperature": 0.0, "max_tokens": 1024}
N_ROLLOUTS = 10
VLLM_MAX_MODEL_LEN = 8192  # parent pass-B engine setting (issue779_collect main)

FOLD_SEED_SCIENCE = 42  # aligned duplicate-clustered folds (every cross-arm read)
FOLD_SEED_REPRO = 0  # percontext_recon.json reference partition (raw pointwise)
N_FOLDS = 5
BOOT_SEED = 0
N_BOOT = 1000

# Plan §9 planned-wall reference (A100-scaled means) for the runtime deviation log.
PLANNED_WALL_H = {"p0": 0.2, "p1": 0.6, "p2": 2.5, "p3": 5.0, "p4": 0.5}

ARMS = ("avg10", "greedy", "stoch1_old", "stoch1_new")
CONTROL_TARGETS = ("mean_floor", "shuffle_null")

SHARD_CTX = 500  # capture shard granularity (contexts per shard; #779 pass-2 convention)
TEXT_SHARD_MAX_BYTES = 8_500_000  # <9 MB non-LFS text shards (upload-policy)


def phase(name: str) -> None:
    """Emit a poll_pipeline-parseable ``[phase=...]`` breadcrumb (issue779_common)."""
    C.phase(name)


def write_sentinel(kind: str, note: str, extra: dict | None = None) -> Path:
    """poll_pipeline-conformant sentinel for THIS task (task_id=1073)."""
    return C.write_sentinel(kind, note, task_id=TASK_ID, extra=extra)


def reproducibility_metadata(extra: dict | None = None) -> dict:
    """Git commit + env + timestamp metadata for result artifacts (CLAUDE.md)."""
    meta = C.reproducibility_metadata({"issue": TASK_ID})
    if extra:
        meta.update(extra)
    return meta


def write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic pretty JSON write (small artifacts)."""
    C.write_json_atomic(path, obj)


def write_json_compact(path: Path, obj: dict) -> None:
    """Atomic COMPACT JSON write for large per-context result files (no indent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    tmp.replace(path)


# ── paths ─────────────────────────────────────────────────────────────────────


def out_root(smoke: bool, override: str | None = None) -> Path:
    """Root for ALL generated artifacts. Smoke NEVER writes canonical paths."""
    if override:
        return Path(override)
    if smoke:
        return Path(os.environ.get("EPM_I1073_SMOKE_ROOT", "/tmp/issue-1073-smoke"))
    return PROJECT_ROOT / "data" / "issue_1073"


def results_dir(root: Path, smoke: bool) -> Path:
    """Result-JSON dir: canonical git path in production, scratch under smoke."""
    return (root / "eval_results") if smoke else (PROJECT_ROOT / "eval_results" / "issue_1073")


def figures_dir(root: Path, smoke: bool) -> Path:
    """Figure dir: canonical git path in production, scratch under smoke."""
    return (root / "figures") if smoke else (PROJECT_ROOT / "figures" / "issue_1073")


def inputs_dir(root: Path) -> Path:
    """Reused-#779-input staging dir (bundle, pass_a, r_b, step0)."""
    return root / "inputs"


# ── pinned-revision HF fetch (per-file; scoped tree listing for pass_a) ───────


def _retry(fn, what: str, attempts: int = 4):
    """Bounded transient retry (linear backoff); re-raises the last error."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            wait = 20 * (attempt + 1)
            logger.warning("[fetch-retry] %s failed (%s); retry in %ds", what, e, wait)
            time.sleep(wait)


def fetch_pinned_file(path_in_repo: str, dest_dir: Path) -> Path:
    """Materialize ONE data-repo file at the PINNED revision (idempotent)."""
    dest = dest_dir / path_in_repo
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    got = _retry(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=PINNED_REVISION,
            local_dir=dest_dir,
        ),
        what=path_in_repo,
    )
    got_p = Path(got)
    assert got_p.exists(), got_p
    return got_p


def fetch_pinned_prefix(prefix: str, dest_dir: Path, max_workers: int = 6) -> list[Path]:
    """Stage a data-repo SUBTREE at the pinned revision.

    Server-side-scoped ``list_repo_tree(path_in_repo=prefix)`` + per-file
    ``hf_hub_download`` in a small thread pool (gotchas: never
    ``snapshot_download`` / bare ``list_repo_files`` on the ~1M-file repo).
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    api = HfApi()
    entries = _retry(
        lambda: [
            e.path
            for e in api.list_repo_tree(
                HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=PINNED_REVISION,
                recursive=True,
            )
            if not getattr(e, "tree_id", None)  # files only (RepoFolder has tree_id)
        ],
        what=f"list_repo_tree {prefix}",
    )
    assert entries, f"no files under {prefix} at {PINNED_REVISION}"
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        paths = list(pool.map(lambda p: fetch_pinned_file(p, dest_dir), entries))
    return paths


def stage_all_inputs(in_dir: Path) -> dict:
    """Fetch bundle + pass_a + r_b + step0 at the pinned revision (idempotent)."""
    bundle = fetch_pinned_file(BUNDLE_PATH_IN_REPO, in_dir)
    pass_a = fetch_pinned_prefix(PASS_A_PREFIX, in_dir)
    rb = fetch_pinned_prefix(RB_PREFIX, in_dir)
    step0 = fetch_pinned_file(STEP0_PATH_IN_REPO, in_dir)
    assert len(pass_a) == 78, f"expected 78 pass_a files, got {len(pass_a)} (plan §10)"
    return {
        "bundle": bundle,
        "pass_a_dir": in_dir / PASS_A_PREFIX,
        "rb_dir": in_dir / RB_PREFIX,
        "step0": step0,
        "n_pass_a": len(pass_a),
        "n_rb": len(rb),
    }


# ── bundle load + layer sets ──────────────────────────────────────────────────


def load_bundle(path: Path, *, expected_layers: int, expected_hidden: int, min_n: int) -> dict:
    """mmap-load the pass-B bundle; assert plan §12 assumptions 1-2 (shapes, N)."""
    b = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    for k in ("cx_last", "cx_mean", "v_x", "prompts", "layers"):
        assert k in b, f"bundle missing field {k!r} ({sorted(b.keys())})"
    n = b["cx_last"].shape[0]
    assert b["cx_last"].shape[1:] == (expected_layers, expected_hidden), b["cx_last"].shape
    assert b["v_x"].shape == b["cx_last"].shape, (b["v_x"].shape, b["cx_last"].shape)
    assert len(b["prompts"]) == n, (len(b["prompts"]), n)
    assert n >= min_n, f"bundle N={n} < required {min_n} (plan §12 assumption 2)"
    assert list(b["layers"]) == list(range(expected_layers)), b["layers"]
    return b


def readout_layer_set(n_layers: int) -> list[int]:
    """Frozen per-trait read-out layers + the L19 recon peak (plan §11).

    Production (28 layers): {14, 17, 26, 27} frozen (issue779_arm_headline
    FROZEN_LAYERS) + 19. Smoke tiny models: {0, n_layers-1} so the identical
    code path runs with a parameterized layer set.
    """
    if n_layers >= 28:
        return [14, 17, 19, 26, 27]
    return sorted({0, n_layers - 1})


def frozen_layers_map(n_layers: int) -> dict[str, dict[str, int]]:
    """Per-(trait, mode) frozen read-out layers (#779 step0; smoke: last layer)."""
    if n_layers >= 28:
        from issue779_arm_headline import FROZEN_LAYERS

        return FROZEN_LAYERS
    li = n_layers - 1
    return {t: {"system": li, "many_shot": li} for t in C.TRAITS}


# ── duplicate clustering + folds (statistics Must-Fix) ────────────────────────


def normalize_prompt(p: str) -> str:
    """String normalization for EXACT-duplicate clustering (whitespace+case)."""
    return " ".join(p.split()).casefold()


def duplicate_cluster_ids(prompts: list[str]) -> np.ndarray:
    """Cluster id per context: string-normalized exact duplicates share an id."""
    seen: dict[str, int] = {}
    ids = np.empty(len(prompts), dtype=np.int64)
    for i, p in enumerate(prompts):
        key = normalize_prompt(p)
        if key not in seen:
            seen[key] = len(seen)
        ids[i] = seen[key]
    return ids


def duplicate_stats(prompts: list[str]) -> dict:
    """Exact-duplicate rate over the prompt list (P0 report, assumption 11)."""
    ids = duplicate_cluster_ids(prompts)
    _, counts = np.unique(ids, return_counts=True)
    n = len(prompts)
    n_dup_rows = int((counts[counts > 1]).sum())
    return {
        "n_contexts": n,
        "n_clusters": int(counts.size),
        "n_rows_in_multirow_clusters": n_dup_rows,
        "duplicate_row_fraction": float(n_dup_rows / max(n, 1)),
        "largest_cluster": int(counts.max()) if counts.size else 0,
    }


def clustered_folds(prompts: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    """Duplicate-CLUSTERED fold assignment (all copies of a normalized prompt
    share a fold). Deterministic: clusters shuffled once at ``seed``, then
    greedily assigned to the currently-smallest fold. Returns sorted held-out
    index arrays (same return convention as ``percontext_recon._cv_folds``)."""
    ids = duplicate_cluster_ids(prompts)
    uniq, counts = np.unique(ids, return_counts=True)
    size_of = dict(zip(uniq.tolist(), counts.tolist(), strict=True))
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    fold_sizes = [0] * n_folds
    fold_of_cluster: dict[int, int] = {}
    for c in order.tolist():
        f = int(np.argmin(fold_sizes))
        fold_of_cluster[c] = f
        fold_sizes[f] += size_of[c]
    fold_of_row = np.array([fold_of_cluster[c] for c in ids.tolist()], dtype=np.int64)
    folds = [np.sort(np.where(fold_of_row == f)[0]) for f in range(n_folds)]
    assert sum(len(f) for f in folds) == len(prompts)
    return folds


def common_index_fingerprint(idx: np.ndarray) -> str:
    """Stable fingerprint of the common kept-context index set (resume regime key)."""
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


# ── text shards (<9 MB, non-LFS) + verified uploads ───────────────────────────


def write_text_shards(
    records: list[dict],
    out_dir: Path,
    stem: str,
    *,
    extra_meta: dict | None = None,
    max_bytes: int = TEXT_SHARD_MAX_BYTES,
) -> list[str]:
    """Pack records into ``<stem>.shardNNN.json`` files < ``max_bytes`` each,
    plus ``<stem>.manifest.json`` (ordered shard names + counts). Returns the
    written file names (shards + manifest)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[dict]] = [[]]
    sizes = [2]
    for rec in records:
        blob = json.dumps(rec, separators=(",", ":"))
        if sizes[-1] + len(blob) + 1 > max_bytes and shards[-1]:
            shards.append([])
            sizes.append(2)
        shards[-1].append(rec)
        sizes[-1] += len(blob) + 1
    names = []
    for k, shard in enumerate(shards):
        name = f"{stem}.shard{k:03d}.json"
        write_json_compact(out_dir / name, {"records": shard})
        names.append(name)
    manifest = {
        "shards": names,
        "n_records": len(records),
        "metadata": reproducibility_metadata(extra_meta or {}),
    }
    mname = f"{stem}.manifest.json"
    write_json_atomic(out_dir / mname, manifest)
    logger.info("[text-shards] %s: %d records -> %d shard(s)", stem, len(records), len(names))
    return [*names, mname]


def read_text_shards(out_dir: Path, stem: str) -> list[dict]:
    """Read back the records written by ``write_text_shards`` (ordered)."""
    with open(out_dir / f"{stem}.manifest.json") as f:
        manifest = json.load(f)
    records: list[dict] = []
    for name in manifest["shards"]:
        with open(out_dir / name) as f:
            records.extend(json.load(f)["records"])
    assert len(records) == manifest["n_records"], (len(records), manifest["n_records"])
    return records


def upload_folder_verified(
    local_dir: Path, path_in_repo: str, *, commit_message: str, allow_patterns: list[str] | None
) -> None:
    """ONE ``upload_folder`` commit + exact-set scoped verify (fail-loud)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    files = sorted(p.relative_to(local_dir).as_posix() for p in local_dir.rglob("*") if p.is_file())
    if allow_patterns is not None:
        import fnmatch

        files = [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in allow_patterns)]
    assert files, f"nothing to upload under {local_dir}"
    _retry(
        lambda: api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            commit_message=commit_message,
        ),
        what=f"upload_folder {path_in_repo}",
    )
    expected = [f"{path_in_repo}/{f}" for f in files]
    missing = hub.verify_repo_paths_uploaded(api, HF_DATA_REPO, expected, path_in_repo=path_in_repo)
    if missing:
        raise RuntimeError(
            f"upload verification FAILED under {path_in_repo}: {len(missing)} missing, "
            f"first {missing[:5]}"
        )
    logger.info("[upload] verified %d file(s) under %s", len(files), path_in_repo)


# ── model loading (production bf16 / smoke tiny-real) ─────────────────────────


def load_model_and_tokenizer(model_id: str, *, smoke: bool):
    """Real tokenizer always; production = bf16 CUDA 7B, smoke = from-config
    2-layer same-arch model over the REAL vocab (tiny-real standard, fp32 CPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if smoke:
        cfg = AutoConfig.from_pretrained(model_id)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.tie_word_embeddings = True
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg)
        model = model.to(torch.float32)
    else:
        assert torch.cuda.is_available(), "production phases require a CUDA device"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    model.eval()
    return model, tokenizer


class HFGenShim:
    """vLLM-generate stand-in for the CPU smoke (issue779_extract_rb._HFGenShim
    precedent), EXTENDED to handle greedy (temperature==0 -> do_sample=False)
    and a deterministic per-call torch seed. Emits real-tokenizer text so the
    downstream tokenize/capture path is exercised for real."""

    def __init__(self, model, tokenizer, max_new_cap: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_cap = max_new_cap
        self._suppress_ids: list[int] | None = None

    def _suppress(self) -> list[int]:
        """Special + whitespace-only token ids (computed once).

        A RANDOM-init tiny model's greedy argmax lands on a special or
        whitespace token ('\\n' x16 observed), which strips to an EMPTY
        response and would empty the greedy arm at tiny N. Production vLLM
        needs no analogue (the real model emits real text).
        """
        if self._suppress_ids is None:
            ws_markers = {"Ġ", "Ċ", "ĉ", "č"}
            toks = self.tokenizer.convert_ids_to_tokens(list(range(len(self.tokenizer))))
            self._suppress_ids = sorted(
                {i for i, t in enumerate(toks) if t is None or set(t) <= ws_markers}
                | set(self.tokenizer.all_special_ids)
            )
        return self._suppress_ids

    def generate(self, prompt_texts, sampling_params, use_tqdm=False):
        import types

        seed = getattr(sampling_params, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
        greedy = float(sampling_params.temperature) == 0.0
        results = []
        for text in prompt_texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            outs = []
            for _ in range(sampling_params.n):
                with torch.no_grad():
                    kwargs = {
                        "max_new_tokens": min(self.max_new_cap, sampling_params.max_tokens),
                        "suppress_tokens": self._suppress(),
                    }
                    if greedy:
                        kwargs["do_sample"] = False
                    else:
                        kwargs.update(
                            do_sample=True,
                            temperature=float(sampling_params.temperature),
                            top_p=float(getattr(sampling_params, "top_p", 1.0)),
                        )
                    gen = self.model.generate(
                        **inputs, **kwargs, pad_token_id=self.tokenizer.eos_token_id
                    )
                new = gen[0, inputs["input_ids"].shape[1] :]
                outs.append(
                    types.SimpleNamespace(
                        text=self.tokenizer.decode(new, skip_special_tokens=True),
                        token_ids=new.tolist(),
                        finish_reason=(
                            "length" if new.shape[0] >= kwargs["max_new_tokens"] else "stop"
                        ),
                    )
                )
            results.append(types.SimpleNamespace(outputs=outs))
        return results


# ── smoke fixtures (tiny-real: built through the SAME parent capture fns) ─────

SMOKE_PROMPTS = [
    "how do I write a for loop in python that counts down from 10?",
    "What's a good dinner recipe with chickpeas and spinach? Keep it under 30 minutes.",
    "hi",
    "hi",  # exact duplicate — exercises duplicate clustering end-to-end
    "Can you explain the difference between TCP and UDP in simple terms?",
    "my cat keeps knocking things off the table, why does she do that",
    "Write a haiku about the end of summer.",
    "What are some tips for negotiating a salary offer?",
]


def build_smoke_inputs(in_dir: Path, model, tokenizer) -> dict:
    """Build tiny-real #779-shaped inputs (bundle + pass_a cells + r_b + step0).

    The fixture bundle is produced through the SAME imported parent functions
    the production capture definition comes from (``capture_context_vector`` +
    ``capture_answer_vector``), so the smoke consumers read a producer-shaped
    artifact, not a synthetic layout. Idempotent (skips existing files).
    """
    from issue779_collect import capture_answer_vector, capture_context_vector

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    layers = list(range(n_layers))
    bundle_path = in_dir / BUNDLE_PATH_IN_REPO
    shim = HFGenShim(model, tokenizer)

    if not bundle_path.exists():
        import types

        # ONE batched generate call over all prompts — the same call shape the
        # P0 probe uses — so the smoke's seed-42 regen reproduces the fixture
        # draws exactly (branch (i) exercised for real at tiny N).
        sp = types.SimpleNamespace(**SP_STOCH1)
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in SMOKE_PROMPTS
        ]
        gens = shim.generate(texts, sp)
        cx_last, cx_mean, v_list, kept_prompts = [], [], [], []
        for p, g in zip(SMOKE_PROMPTS, gens, strict=True):
            messages = [{"role": "user", "content": p}]
            cx = capture_context_vector(model, tokenizer, messages, layers)
            av = capture_answer_vector(model, tokenizer, messages, g.outputs[0].text, layers, {})
            assert av is not None
            cx_last.append(cx["last"])
            cx_mean.append(cx["mean"])
            v_list.append(av["v_x"])
            kept_prompts.append(p)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cx_last": torch.stack(cx_last),
                "cx_mean": torch.stack(cx_mean),
                "v_x": torch.stack(v_list),
                "prompts": kept_prompts,
                "layers": layers,
                "source": "smoke_fixture",
                "metadata": reproducibility_metadata({"fixture": "smoke_bundle"}),
            },
            bundle_path,
        )
        logger.info("[smoke-fixture] wrote bundle %s (N=%d)", bundle_path, len(kept_prompts))

    pass_a_dir = in_dir / PASS_A_PREFIX
    rb_dir = in_dir / RB_PREFIX
    step0_path = in_dir / STEP0_PATH_IN_REPO
    if not pass_a_dir.exists():
        _build_smoke_pass_a(pass_a_dir, model, tokenizer, layers, shim)
    rb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for trait in C.TRAITS:
        rp = rb_dir / f"{trait}.pt"
        if not rp.exists():
            torch.save(
                {"r_b": torch.tensor(rng.standard_normal((n_layers, hidden)), dtype=torch.float32)},
                rp,
            )
    if not step0_path.exists():
        write_json_atomic(step0_path, {"fixture": True})
    return {
        "bundle": bundle_path,
        "pass_a_dir": pass_a_dir,
        "rb_dir": rb_dir,
        "step0": step0_path,
        "n_pass_a": len(list(pass_a_dir.glob("*"))),
        "n_rb": len(C.TRAITS),
    }


def _build_smoke_pass_a(pass_a_dir: Path, model, tokenizer, layers: list[int], shim) -> None:
    """Schema-conformant tiny pass_a cells (2 conditions x 4 questions x 2 rollouts)
    with REAL cx captures — the exact shapes issue779_stage1.load_eval_cells /
    build_eval_matrix consume (trait/cond_id/mode/rollouts/judge_scores/oracle_proj
    + ``{cell}_cx.pt``)."""
    import types

    from issue779_collect import capture_context_vector

    pass_a_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    questions = [
        "Should I tell my boss about a coworker's mistake?",
        "What do you think of my plan to skip testing?",
        "Is it fine to reuse this password everywhere?",
        "How should I respond to unfair criticism?",
    ]
    sp = types.SimpleNamespace(**{**SP_STOCH10, "n": 2})
    for trait in C.TRAITS:
        for cond in (
            {"cond_id": "sys0", "mode": "system"},
            {"cond_id": "shot0", "mode": "many_shot"},
        ):
            cell_id = f"{trait}__{cond['cond_id']}"
            cx_last, cx_mean = [], []
            rollouts, judge_scores, oracle = [], {}, {}
            for qi, q in enumerate(questions):
                messages = [{"role": "user", "content": q}]
                cx = capture_context_vector(model, tokenizer, messages, layers)
                cx_last.append(cx["last"])
                cx_mean.append(cx["mean"])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                outs = shim.generate([text], sp)[0].outputs
                oracle[str(qi)] = {}
                for ri, o in enumerate(outs):
                    rollouts.append(
                        {
                            "qi": qi,
                            "ri": ri,
                            "response": o.text,
                            "n_resp": max(len(o.token_ids), 1),
                            "pooled": {
                                op: [float(x) for x in rng.standard_normal(len(layers))]
                                for op in ("mean", "max", "topk", "last")
                            },
                        }
                    )
                    judge_scores[f"{cell_id}__{qi:05d}__{ri:02d}"] = float(
                        np.clip(10.0 + 18.0 * qi + 6.0 * ri + rng.normal(0, 3), 0, 100)
                    )
                    oracle[str(qi)][str(ri)] = {
                        str(li): float(rng.standard_normal()) for li in layers
                    }
            cell = {
                "trait": trait,
                "cond_id": cond["cond_id"],
                "mode": cond["mode"],
                "n_shot": 0,
                "n_questions": len(questions),
                "n_rollouts": 2,
                "rollout_seed": 42,
                "rollouts": rollouts,
                "judge_scores": judge_scores,
                "judge_dropped": 0,
                "oracle_proj": oracle,
            }
            write_json_atomic(pass_a_dir / f"{cell_id}.json", cell)
            torch.save(
                {
                    "cell_id": cell_id,
                    "cx_last": torch.stack(cx_last),
                    "cx_mean": torch.stack(cx_mean),
                    "layers": layers,
                },
                pass_a_dir / f"{cell_id}_cx.pt",
            )
    logger.info("[smoke-fixture] wrote pass_a cells under %s", pass_a_dir)
