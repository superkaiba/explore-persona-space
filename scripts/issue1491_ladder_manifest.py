"""Phase-0 ladder manifest builder for task #1491.

Deterministically (seed 42) builds the ladder context set at
``superkaiba1/explore-persona-space-data:issue1491_scale_ladder/manifest/``,
derived from the parent's #779 n1M sampling manifest.

Per plan §4.1 (task #1491, v4, approved 2026-08-03), the manifest contains:

- ``train_25k`` — 25,000 LMSYS rows sampled without replacement from the
  525,485 LMSYS entries of the parent manifest (seed 42).
- ``val_400`` + ``test_1000`` — the pinned #779 split, deterministically
  re-derived via ``sample_disjoint_n50k(...)['round1']`` +
  ``_valtest_prompts_from_round1``; asserted to hash to the committed
  ``val_sha256`` / ``test_sha256`` values from
  ``eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json``.
- ``wc_test_1k`` — 1,000 WildChat rows sampled from the 434,515 WildChat
  entries of the parent manifest (disjoint from ``train_25k`` by corpus).
- ``tierB_3600`` — the first 3,600 ids of ``train_25k`` in sampled order.
- ``overlength_skip`` — the sidecar recording rows dropped by the
  over-length filter (rendered-token budget ≤ 7104 = 8192 − 1024 − 64,
  parent commit ``bd9f6865de``); applied ONCE at manifest build.

Also asserted: ``chat_template`` + ``vocab_hash`` equality across all six
``tokenizer_config.json`` files for ``Qwen/Qwen2.5-{0.5B, 1.5B, 3B, 7B,
14B, 32B}-Instruct`` — so the over-length skip set stays identical across
scales and the context set is matched by construction.

Port source (recorded in ``epm:progress v6`` on task #1491):
``port_source: origin/main`` for all parent-branch modules. Helpers are
imported by module path; no vendoring.

Runtime: CPU-only. Uses ``HF_TOKEN`` from ``.env`` via the project's
dotenv loader. Uploads one ``upload_folder`` commit (Upload Policy). The
val/test re-derivation streams LMSYS live via ``sample_disjoint_n50k`` —
production wall-time ~20-40 min (rate-limited HF stream).

CLI modes:
- Default: build + upload if the HF prefix is empty; verify + no-op if
  already populated.
- ``--dry-run``: verify state (tokenizer identity + parent manifest sha)
  without downloading corpora or uploading.
- ``--force``: re-upload even if the manifest verifies present.
- ``--smoke``: use a small synthetic stream (100 rows) instead of live
  LMSYS; produces a smoke-tagged local manifest, never uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# The port-source decision (epm:progress v6) mandates imports from origin/main
# copies — no vendoring. Signature-smoke recorded there.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from issue779_ffc_n1m_generate_capture import (  # type: ignore  # noqa: E402
    _filter_overlength_prompts,
    _valtest_prompts_from_round1,
    N_ROUND1,
)
from issue779_ffc_n50k_generate_capture import sample_disjoint_n50k  # type: ignore  # noqa: E402

# fixed_split lives in the fair-comparison module referenced by n1m via
# ``import issue779_fitter_fair_comparison as F``; we import the same way.
import issue779_fitter_fair_comparison as F  # type: ignore  # noqa: E402

logger = logging.getLogger("issue1491_ladder_manifest")

# ---------------------------------------------------------------------------
# Constants (grounded on plan v4 + committed n1m_fits.json + parent meta.json)
# ---------------------------------------------------------------------------

PARENT_MANIFEST_HF_REPO = "superkaiba1/explore-persona-space-data"
PARENT_MANIFEST_HF_PATH = "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
PARENT_NEW_PROMPT_SHA256 = "2b14762a15d316c602332a749ebd87c733d687d4165eb5d0038c298e0d27ce46"

# Ladder output — child-issue prefix; NEVER the parent's (Upload Policy /
# runtime-reuse clobber clause, plan §10 item (i)).
LADDER_HF_REPO = "superkaiba1/explore-persona-space-data"
LADDER_HF_PREFIX = "issue1491_scale_ladder/manifest"

# Split sizes + parent-anchor pinned val/test SHAs (from
# eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json .split @ 507bf182f6).
N_TRAIN_25K = 25_000
N_VAL_400 = 400
N_TEST_1000 = 1000
N_WC_TEST_1K = 1000
N_TIERB_3600 = 3600
VAL_SHA256 = "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
TEST_SHA256 = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"

# fixed_split arguments (parent-anchor): 5000 round-1 rows partitioned into
# 3600 train + 400 val + 1000 test with seed 42.
FIXED_SPLIT_N = 5000
FIXED_SPLIT_TRAIN = 3600
FIXED_SPLIT_VAL = 400
FIXED_SPLIT_TEST = 1000
FIXED_SPLIT_SEED = 42

# Over-length filter budget (parent commit bd9f6865de).
OVERLENGTH_MAX_MODEL_LEN = 8192
OVERLENGTH_RESERVE_GEN = 1024
OVERLENGTH_RESERVE_SLACK = 64
OVERLENGTH_BUDGET = (
    OVERLENGTH_MAX_MODEL_LEN - OVERLENGTH_RESERVE_GEN - OVERLENGTH_RESERVE_SLACK
)  # = 7104

# Qwen2.5 scale ladder — all six Instruct sizes.
QWEN_LADDER = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

# The chat template used for over-length filtering; identical across ladder
# per the tokenizer-identity assert. We render via any one size (7B, the
# anchor) after asserting equality.
QWEN_OVERLENGTH_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"

# Parent's manifest expected shape (from meta.json).
PARENT_MANIFEST_N_LMSYS = 525_485
PARENT_MANIFEST_N_WILDCHAT = 434_515
PARENT_MANIFEST_N_NEW = 960_000

# Deterministic sampling seeds.
SEED_TRAIN_25K = 42
SEED_WC_TEST_1K = 42

# Manifest files this script writes.
MANIFEST_META_NAME = "meta.json"
SPLIT_FILES = {
    "train_25k": "train_25k.jsonl",
    "val_400": "val_400.jsonl",
    "test_1000": "test_1000.jsonl",
    "wc_test_1k": "wc_test_1k.jsonl",
    "tierB_3600": "tierB_3600.jsonl",
    "overlength_skip": "overlength_skip.jsonl",
}


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env for HF_TOKEN — used by huggingface_hub calls."""
    try:
        from explore_persona_space.orchestrate.env import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:  # pragma: no cover - fallback if the helper moves
        from dotenv import load_dotenv as _ld  # type: ignore

        _ld()


def _hf_api():
    """Return an HfApi instance; env must be loaded first."""
    from huggingface_hub import HfApi  # type: ignore

    return HfApi()


def _iter_parent_rows(cache_dir: Path) -> Iterable[dict]:
    """Download and stream the parent manifest's 87 part_NNNNN.jsonl files.

    Yields row dicts in manifest order. Downloads each file once into
    ``cache_dir`` and reads incrementally.
    """
    from huggingface_hub import hf_hub_download  # type: ignore

    api = _hf_api()
    files = list(
        api.list_repo_tree(
            repo_id=PARENT_MANIFEST_HF_REPO,
            path_in_repo=PARENT_MANIFEST_HF_PATH,
            repo_type="dataset",
            recursive=True,
        )
    )
    parts = sorted(f.path for f in files if f.path.endswith(".jsonl"))
    if not parts:
        raise RuntimeError(
            f"parent manifest at {PARENT_MANIFEST_HF_REPO}:{PARENT_MANIFEST_HF_PATH} "
            "contains no .jsonl parts"
        )
    logger.info("streaming %d parts from parent manifest", len(parts))
    for part in parts:
        local = hf_hub_download(
            repo_id=PARENT_MANIFEST_HF_REPO,
            filename=part,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _verify_parent_meta(cache_dir: Path) -> dict:
    """Fetch + verify the parent manifest's meta.json."""
    from huggingface_hub import hf_hub_download  # type: ignore

    local = hf_hub_download(
        repo_id=PARENT_MANIFEST_HF_REPO,
        filename=f"{PARENT_MANIFEST_HF_PATH}/{MANIFEST_META_NAME}",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    with open(local, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("new_prompt_sha256") == PARENT_NEW_PROMPT_SHA256, (
        f"parent manifest prompt-set sha drifted: got "
        f"{meta.get('new_prompt_sha256')!r}, expected {PARENT_NEW_PROMPT_SHA256!r}"
    )
    assert meta.get("n_lmsys") == PARENT_MANIFEST_N_LMSYS, (
        f"parent n_lmsys drifted: {meta.get('n_lmsys')} != {PARENT_MANIFEST_N_LMSYS}"
    )
    assert meta.get("n_wildchat") == PARENT_MANIFEST_N_WILDCHAT, (
        f"parent n_wildchat drifted: {meta.get('n_wildchat')} != {PARENT_MANIFEST_N_WILDCHAT}"
    )
    assert meta.get("n_new") == PARENT_MANIFEST_N_NEW, (
        f"parent n_new drifted: {meta.get('n_new')} != {PARENT_MANIFEST_N_NEW}"
    )
    return meta


def _verify_ladder_manifest_present() -> bool:
    """Return True iff the ladder manifest already exists on HF with the
    expected file set."""
    api = _hf_api()
    try:
        entries = list(
            api.list_repo_tree(
                repo_id=LADDER_HF_REPO,
                path_in_repo=LADDER_HF_PREFIX,
                repo_type="dataset",
                recursive=True,
            )
        )
    except Exception:
        return False
    present = {e.path.split("/")[-1] for e in entries if not e.path.endswith("/")}
    expected = {MANIFEST_META_NAME, *SPLIT_FILES.values()}
    return expected.issubset(present)


# ---------------------------------------------------------------------------
# Tokenizer identity assert (chat_template + vocab hash across the 6 sizes)
# ---------------------------------------------------------------------------


def _tokenizer_identity_assert() -> dict:
    """Assert chat_template + vocab hash equality across the 6 Qwen2.5 sizes.

    Returns a dict with the shared chat_template hash, vocab hash, and per-model
    ``tokenizer_config.json`` shas (for the manifest).
    """
    from huggingface_hub import hf_hub_download  # type: ignore

    per_model = []
    templates: set[str] = set()
    vocab_hashes: set[str] = set()
    tokenizer_config_hashes: dict[str, str] = {}
    for model_id in QWEN_LADDER:
        cfg_local = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
        with open(cfg_local, "rb") as fh:
            cfg_bytes = fh.read()
        cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
        tokenizer_config_hashes[model_id] = cfg_sha
        cfg = json.loads(cfg_bytes)
        chat_template = cfg.get("chat_template")
        assert chat_template, f"{model_id}: missing chat_template"
        templates.add(chat_template)
        vocab_local = hf_hub_download(repo_id=model_id, filename="vocab.json")
        with open(vocab_local, "rb") as fh:
            vocab_bytes = fh.read()
        vocab_hash = hashlib.sha256(vocab_bytes).hexdigest()
        vocab_hashes.add(vocab_hash)
        per_model.append(
            {
                "model_id": model_id,
                "tokenizer_config_sha256": cfg_sha,
                "vocab_sha256": vocab_hash,
            }
        )
    assert len(templates) == 1, (
        f"chat_template mismatch across ladder — {len(templates)} distinct templates"
    )
    assert len(vocab_hashes) == 1, (
        f"vocab hash mismatch across ladder — {len(vocab_hashes)} distinct vocabs"
    )
    chat_template = templates.pop()
    return {
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "vocab_sha256": next(iter(vocab_hashes)),
        "per_model": per_model,
        "tokenizer_config_hashes": tokenizer_config_hashes,
    }


# ---------------------------------------------------------------------------
# Over-length filter (chat-template-aware token count, per parent commit
# bd9f6865de — shared budget 7104 across the ladder by construction).
# ---------------------------------------------------------------------------


def _make_token_len_fn():
    """Return ``token_len_fn(prompt: str) -> int`` under the shared template."""
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(QWEN_OVERLENGTH_TOKENIZER)

    def token_len_fn(prompt: str) -> int:
        msgs = [{"role": "user", "content": prompt}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(rendered, add_special_tokens=False)["input_ids"])

    return token_len_fn


# ---------------------------------------------------------------------------
# Sampling + val/test re-derivation
# ---------------------------------------------------------------------------


def _sample_positions(pool_size: int, k: int, seed: int) -> list[int]:
    """Deterministic sample of ``k`` positions from ``range(pool_size)``."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(pool_size, size=k, replace=False)
    return sorted(int(p) for p in picks)


def _derive_valtest_prompts(
    *, smoke: bool, smoke_stream: list[dict] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Return (round1, val, test) per the parent's deterministic path.

    In production mode this STREAMS LMSYS live (rate-limited; ~20-40 min).
    In smoke mode a caller-provided synthetic stream is used.
    """
    if smoke:
        assert smoke_stream is not None, "smoke=True requires smoke_stream"
        result = sample_disjoint_n50k(
            skip_round1=N_ROUND1, n_n10k=0, n_new=1, stream_iter=iter(smoke_stream)
        )
    else:
        result = sample_disjoint_n50k(skip_round1=N_ROUND1, n_n10k=0, n_new=1)
    round1 = list(result["round1"])
    assert len(round1) == N_ROUND1, (
        f"sample_disjoint_n50k returned {len(round1)} round-1 rows, expected {N_ROUND1}"
    )
    valtest = _valtest_prompts_from_round1(round1, check_ctx0=not smoke)
    # fixed_split partitions the 5000 round-1 into (train_3600, val_400, test_1000).
    _r1_train, val, test = F.fixed_split(
        FIXED_SPLIT_N,
        FIXED_SPLIT_TRAIN,
        FIXED_SPLIT_VAL,
        FIXED_SPLIT_TEST,
        FIXED_SPLIT_SEED,
    )
    # _valtest_prompts_from_round1 returns the 1400 pinned val+test prompt strings
    # in row order; slice against the fixed_split indices to recover val/test.
    # Convention: fixed_split returns train_indices, val_indices, test_indices.
    val_prompts = [round1[i] for i in val]
    test_prompts = [round1[i] for i in test]
    # Sanity: valtest should equal val_prompts + test_prompts (contains 1400 rows).
    assert len(valtest) == len(val_prompts) + len(test_prompts), (
        f"valtest={len(valtest)} != val+test={len(val_prompts) + len(test_prompts)}"
    )
    return round1, val_prompts, test_prompts


def _sha256_prompt_list(prompts: list[str]) -> str:
    """SHA256 of a list of prompts, canonicalized as newline-joined UTF-8."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build + write manifest
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )


def build_manifest(out_dir: Path, *, smoke: bool = False, dry_run: bool = False) -> dict:
    """Deterministically build the ladder manifest.

    Returns the meta dict; writes files into ``out_dir``.
    """
    _load_env()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer identity assert first (cheap; also covered by dry-run).
    logger.info("asserting chat_template + vocab equality across the 6 Qwen sizes")
    tokenizer_info = _tokenizer_identity_assert()

    if dry_run:
        # In --dry-run we skip corpus downloads + upload; assert tokenizer
        # identity + parent manifest sha only.
        cache_dir = out_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        parent_meta = _verify_parent_meta(cache_dir)
        meta = {
            "mode": "dry-run",
            "recipe_version": "issue1491_ladder-v1",
            "parent_manifest": {
                "repo": PARENT_MANIFEST_HF_REPO,
                "path": PARENT_MANIFEST_HF_PATH,
                "new_prompt_sha256": parent_meta["new_prompt_sha256"],
                "n_lmsys": parent_meta["n_lmsys"],
                "n_wildchat": parent_meta["n_wildchat"],
            },
            "tokenizer": tokenizer_info,
        }
        (out_dir / "dry_run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("dry-run OK — tokenizer + parent-manifest assertions pass")
        return meta

    # 2. Fetch parent manifest + build pools.
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    parent_meta = _verify_parent_meta(cache_dir)

    if smoke:
        # Synthetic 100-row pool: alternating lmsys/wildchat, deterministic.
        lmsys_rows = [
            {"prompt": f"smoke lmsys q{i}", "corpus": "lmsys", "i": i, "stream_pos": i}
            for i in range(50)
        ]
        wildchat_rows = [
            {
                "prompt": f"smoke wc q{i}",
                "corpus": "wildchat",
                "i": 50 + i,
                "stream_pos": 50 + i,
            }
            for i in range(50)
        ]
    else:
        logger.info("materializing parent manifest rows (960k rows, ~200 MB in memory)")
        lmsys_rows: list[dict] = []
        wildchat_rows: list[dict] = []
        for row in _iter_parent_rows(cache_dir):
            if row["corpus"] == "lmsys":
                lmsys_rows.append(row)
            elif row["corpus"] == "wildchat":
                wildchat_rows.append(row)
        assert len(lmsys_rows) == PARENT_MANIFEST_N_LMSYS, (
            f"lmsys row count mismatch: {len(lmsys_rows)} != {PARENT_MANIFEST_N_LMSYS}"
        )
        assert len(wildchat_rows) == PARENT_MANIFEST_N_WILDCHAT, (
            f"wildchat row count mismatch: {len(wildchat_rows)} != {PARENT_MANIFEST_N_WILDCHAT}"
        )

    # 3. Sample train_25k + wc_test_1k.
    if smoke:
        n_train = min(N_TRAIN_25K, len(lmsys_rows))
        n_wc = min(N_WC_TEST_1K, len(wildchat_rows))
    else:
        n_train = N_TRAIN_25K
        n_wc = N_WC_TEST_1K
    train_positions = _sample_positions(len(lmsys_rows), n_train, SEED_TRAIN_25K)
    wc_positions = _sample_positions(len(wildchat_rows), n_wc, SEED_WC_TEST_1K)
    train_25k = [lmsys_rows[p] for p in train_positions]
    wc_test_1k = [wildchat_rows[p] for p in wc_positions]

    # 4. Re-derive val/test.
    if smoke:
        # A synthetic round1 satisfying N_ROUND1: reuse train prompts to bulk it out.
        smoke_stream = [
            {"conversation": [{"role": "user", "content": f"smoke r1 q{i}"}]}
            for i in range(N_ROUND1 + 10)
        ]
        _, val_prompts, test_prompts = _derive_valtest_prompts(
            smoke=True, smoke_stream=smoke_stream
        )
    else:
        _, val_prompts, test_prompts = _derive_valtest_prompts(smoke=False)

    assert len(val_prompts) == N_VAL_400, f"val_400 size wrong: {len(val_prompts)}"
    assert len(test_prompts) == N_TEST_1000, f"test_1000 size wrong: {len(test_prompts)}"
    # SHA anchor asserts (production only; smoke uses synthetic streams).
    if not smoke:
        val_sha = _sha256_prompt_list(val_prompts)
        test_sha = _sha256_prompt_list(test_prompts)
        assert val_sha == VAL_SHA256, f"val_sha256 drift: got {val_sha}, expected {VAL_SHA256}"
        assert test_sha == TEST_SHA256, f"test_sha256 drift: got {test_sha}, expected {TEST_SHA256}"
    val_rows = [{"prompt": p, "corpus": "lmsys", "split": "val"} for p in val_prompts]
    test_rows = [{"prompt": p, "corpus": "lmsys", "split": "test"} for p in test_prompts]

    # 5. tierB_3600 = first 3600 of train_25k in sampled order.
    tierb_3600 = train_25k[: min(N_TIERB_3600, len(train_25k))]

    # 6. Over-length filter (shared budget 7104; single pass over all splits).
    logger.info("applying over-length filter (budget=%d)", OVERLENGTH_BUDGET)
    token_len_fn = _make_token_len_fn()

    def _filter(rows: list[dict], split: str) -> tuple[list[dict], list[dict]]:
        """Return (kept, skipped) for a split."""
        prompts = [r["prompt"] for r in rows]
        cis = list(range(len(prompts)))
        # _filter_overlength_prompts returns (kept_prompts, kept_cis, skipped_dicts).
        # We only need the kept_cis (local indices that survived the budget) —
        # the parent helper is deterministic + order-preserving, so kept_cis is
        # a subset of the input `cis` (= range(len(prompts))).
        _, kept_cis, _ = _filter_overlength_prompts(prompts, cis, token_len_fn, OVERLENGTH_BUDGET)
        kept_ids = set(kept_cis)
        kept, skipped = [], []
        for local_i, r in enumerate(rows):
            enriched = {**r, "split": split, "ladder_local_id": local_i}
            if local_i in kept_ids:
                kept.append(enriched)
            else:
                skipped.append(enriched)
        return kept, skipped

    train_kept, train_skip = _filter(train_25k, "train_25k")
    val_kept, val_skip = _filter(val_rows, "val_400")
    test_kept, test_skip = _filter(test_rows, "test_1000")
    wc_kept, wc_skip = _filter(wc_test_1k, "wc_test_1k")
    # tierB is a subset of train_25k; propagate the skip mask.
    train_kept_ids = {r["ladder_local_id"] for r in train_kept}
    tierb_kept = [
        {**r, "split": "tierB_3600"} for i, r in enumerate(tierb_3600) if i in train_kept_ids
    ]

    # 7. Write jsonl files.
    _write_jsonl(out_dir / SPLIT_FILES["train_25k"], train_kept)
    _write_jsonl(out_dir / SPLIT_FILES["val_400"], val_kept)
    _write_jsonl(out_dir / SPLIT_FILES["test_1000"], test_kept)
    _write_jsonl(out_dir / SPLIT_FILES["wc_test_1k"], wc_kept)
    _write_jsonl(out_dir / SPLIT_FILES["tierB_3600"], tierb_kept)
    _write_jsonl(
        out_dir / SPLIT_FILES["overlength_skip"],
        train_skip + val_skip + test_skip + wc_skip,
    )

    # 8. Meta.
    meta = {
        "mode": "smoke" if smoke else "production",
        "recipe_version": "issue1491_ladder-v1",
        "seed": {"train_25k": SEED_TRAIN_25K, "wc_test_1k": SEED_WC_TEST_1K},
        "parent_manifest": {
            "repo": PARENT_MANIFEST_HF_REPO,
            "path": PARENT_MANIFEST_HF_PATH,
            "new_prompt_sha256": parent_meta["new_prompt_sha256"],
            "n_lmsys": parent_meta["n_lmsys"],
            "n_wildchat": parent_meta["n_wildchat"],
        },
        "tokenizer": tokenizer_info,
        "overlength_filter": {
            "max_model_len": OVERLENGTH_MAX_MODEL_LEN,
            "reserve_gen": OVERLENGTH_RESERVE_GEN,
            "reserve_slack": OVERLENGTH_RESERVE_SLACK,
            "budget": OVERLENGTH_BUDGET,
            "parent_commit_sha": "bd9f6865de",
        },
        "splits": {
            "train_25k": {
                "kept": len(train_kept),
                "skipped": len(train_skip),
                "declared": N_TRAIN_25K,
            },
            "val_400": {
                "kept": len(val_kept),
                "skipped": len(val_skip),
                "declared": N_VAL_400,
                "sha256": VAL_SHA256,
            },
            "test_1000": {
                "kept": len(test_kept),
                "skipped": len(test_skip),
                "declared": N_TEST_1000,
                "sha256": TEST_SHA256,
            },
            "wc_test_1k": {
                "kept": len(wc_kept),
                "skipped": len(wc_skip),
                "declared": N_WC_TEST_1K,
            },
            "tierB_3600": {
                "kept": len(tierb_kept),
                "declared": N_TIERB_3600,
            },
        },
        "output_files": SPLIT_FILES,
    }
    (out_dir / MANIFEST_META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def upload_manifest(local_dir: Path) -> None:
    """Upload the manifest tree as ONE ``upload_folder`` commit."""
    api = _hf_api()
    logger.info(
        "uploading %s -> %s:%s",
        local_dir,
        LADDER_HF_REPO,
        LADDER_HF_PREFIX,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=LADDER_HF_PREFIX,
        repo_id=LADDER_HF_REPO,
        repo_type="dataset",
        commit_message=(
            f"issue1491 ladder manifest v1 (train_25k+val+test+wc+tierB; "
            f"budget={OVERLENGTH_BUDGET})"
        ),
        allow_patterns=list(SPLIT_FILES.values()) + [MANIFEST_META_NAME],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task #1491 Phase-0 ladder manifest builder.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/tmp")) / "issue1491_ladder_manifest",
        help="Local staging directory (default: $HF_HOME/issue1491_ladder_manifest).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify tokenizer identity + parent-manifest sha; write nothing to HF.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if the ladder manifest is already present on HF.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use synthetic 100-row corpora instead of live LMSYS/WildChat streams; "
        "produces a smoke-tagged local manifest, never uploaded.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    _load_env()

    if args.dry_run:
        meta = build_manifest(args.out_dir, smoke=False, dry_run=True)
        print(f"DRY-RUN OK — wrote {args.out_dir / 'dry_run_meta.json'}")
        print(json.dumps(meta, indent=2, ensure_ascii=False)[:400])
        return 0

    if args.smoke:
        smoke_dir = args.out_dir.with_name(args.out_dir.name + "_smoke")
        meta = build_manifest(smoke_dir, smoke=True)
        print(f"SMOKE OK — wrote {smoke_dir}/meta.json")
        print(json.dumps(meta["splits"], indent=2))
        return 0

    if not args.force and _verify_ladder_manifest_present():
        print(
            "VERIFY OK — ladder manifest already present on HF at "
            f"{LADDER_HF_REPO}:{LADDER_HF_PREFIX} (use --force to re-upload)"
        )
        return 0

    meta = build_manifest(args.out_dir, smoke=False)
    upload_manifest(args.out_dir)
    print(f"UPLOAD OK — wrote and uploaded {LADDER_HF_REPO}:{LADDER_HF_PREFIX}")
    print(json.dumps(meta["splits"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
