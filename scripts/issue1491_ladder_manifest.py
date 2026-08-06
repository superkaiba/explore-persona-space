"""Phase-0 ladder manifest builder for task #1491.

Deterministically (seed 42) builds the ladder context set at
``superkaiba1/explore-persona-space-data:issue1491_scale_ladder/manifest/``,
derived from the parent's #779 n1M sampling manifest.

Per plan §4.1 (task #1491, v4, approved 2026-08-03), the manifest contains:

- ``train_25k`` — 25,000 LMSYS rows sampled without replacement from the
  525,485 LMSYS entries of the parent manifest (seed 42).
- ``val_400`` + ``test_1000`` — the pinned #779 split, deterministically
  re-derived via ``sample_disjoint_n50k(...)['round1']`` +
  ``_valtest_prompts_from_round1``; membership confirmed in THREE frozen
  domains (``_assert_pinned_membership`` — round-1 prompt sha, fixed_split
  INDEX pins, val/test prompt digests; the #1776 three-domain port. The
  committed ``n1m_fits.json`` ``val_sha256``/``test_sha256`` values are
  INDEX-array digests, never compared against prompt hashes).
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

# NOTE: numpy is deliberately NOT imported at module top — see _sample_positions.
# This module loads .env inside _load_env() rather than at module scope, so a
# module-top heavy import would run before the shared-VM thread caps (#847) bind.

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
from issue779_ffc_n50k_generate_capture import (  # type: ignore  # noqa: E402
    _sha_ids_or_prompts,
    sample_disjoint_n50k,
)

# The INDEX-pin source for the three-domain membership check (P0 crash fix —
# port of issue1776_contexts.py @ 04ce114b8fb2). Module-top is safe: the n1m
# import above already runs load_dotenv() before any torch import (#847).
import issue779_ffc_n50k_fits as N50F  # type: ignore  # noqa: E402

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

# Split sizes.
N_TRAIN_25K = 25_000
N_VAL_400 = 400
N_TEST_1000 = 1000
N_WC_TEST_1K = 1000
N_TIERB_3600 = 3600

# ── Pinned membership shas — THREE frozen domains (P0 crash fix, #1776 port) ──
#
# A sha pin lives in a DOMAIN (.claude/rules/gotchas.md). The two INDEX pins
# below are sha256 digests of the ORIGINAL #779 round's
# fixed_split(5000, 3600, 400, 1000, 42) int64 INDEX arrays (F._sha_ids — the
# domain of N50F._pinned_original_shas / the committed fair_comparison.json).
# They are NOT prompt-string digests: the P0 production crash (pod-1491,
# att rp-20260805T043306Z) asserted a PROMPT digest against them, a compare
# that can never pass on any input. Compare them only against index digests.
VAL_400_INDEX_SHA = "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
TEST_1000_INDEX_SHA = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"
# Frozen round-1 PROMPT-membership sha — N10._sha_prompts / N50._sha_ids_or_prompts
# (b"\x00"-separated) over the 5,000 round-1 first-turns = the #779 n1m sampling
# manifest's used_shas.round1. A live re-stream reproducing it holds EXACTLY the
# pinned membership — the REAL LMSYS stream-drift guard.
ROUND1_PROMPT_SHA = "d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805"
# Derived prompt-list digests of the pinned val-400 / test-1000 (round1[idx]
# under the pinned split), frozen 2026-07-29 by #1776 from a VM re-stream whose
# round-1 sha matched ROUND1_PROMPT_SHA and whose recomputed split-index shas
# matched the INDEX pins — the tertiary composition check.
VAL_400_PROMPT_SHA = "e8c8beb0fed383674c08e19cb6d9a56ca781d5182ba77cab138af33c06aed738"
TEST_1000_PROMPT_SHA = "bb60a2827bdc11675699414cda787c9be8ad3b836e9f529a528dc59a6726d9ef"

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
    # NO bare-dotenv fallback. The project wrapper is the required loader
    # (`workflow_lint.py --check-dotenv-before-hf-import`), and the previous
    # `except Exception:` fallback swallowed genuine project-loader faults —
    # then loaded a DIFFERENT env, so an HF_TOKEN problem surfaced later as an
    # opaque auth error instead of here. If the helper moves, this fails loud.
    from explore_persona_space.orchestrate.env import load_dotenv  # type: ignore

    load_dotenv()


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

    from explore_persona_space.orchestrate import hub  # type: ignore

    api = _hf_api()
    # list_repo_tree is LAZY — materialize the list INSIDE the thunk so the
    # retry covers iteration, not just the call (the #1482 429-storm class).
    files = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: whole listing wrapped in hub.retry_transient here
            api.list_repo_tree(
                repo_id=PARENT_MANIFEST_HF_REPO,
                path_in_repo=PARENT_MANIFEST_HF_PATH,
                repo_type="dataset",
                recursive=True,
            )
        ),
        what=f"list_repo_tree({PARENT_MANIFEST_HF_PATH})",
    )
    parts = sorted(f.path for f in files if f.path.endswith(".jsonl"))
    if not parts:
        raise RuntimeError(
            f"parent manifest at {PARENT_MANIFEST_HF_REPO}:{PARENT_MANIFEST_HF_PATH} "
            "contains no .jsonl parts"
        )
    logger.info("streaming %d parts from parent manifest", len(parts))
    for part in parts:
        local = hub.retry_transient(
            lambda part=part: hf_hub_download(
                repo_id=PARENT_MANIFEST_HF_REPO,
                filename=part,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            ),
            what=f"hf_hub_download({part})",
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

    from explore_persona_space.orchestrate import hub  # type: ignore

    local = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=PARENT_MANIFEST_HF_REPO,
            filename=f"{PARENT_MANIFEST_HF_PATH}/{MANIFEST_META_NAME}",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        ),
        what=f"hf_hub_download({MANIFEST_META_NAME})",
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
    from huggingface_hub.errors import (  # type: ignore
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    from explore_persona_space.orchestrate import hub  # type: ignore

    api = _hf_api()
    try:
        # Retry the transient class BEFORE deciding absence. The narrowing
        # below correctly stopped swallowing faults, but without a retry a
        # single 429/5xx blip then CRASHES Phase 0 outright. retry_transient
        # re-raises 404s unchanged, so the absence semantics below survive.
        # list_repo_tree is LAZY — materialize inside the thunk so the retry
        # covers iteration too (#1482).
        entries = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: whole listing wrapped in hub.retry_transient here
                api.list_repo_tree(
                    repo_id=LADDER_HF_REPO,
                    path_in_repo=LADDER_HF_PREFIX,
                    repo_type="dataset",
                    recursive=True,
                )
            ),
            what=f"list_repo_tree({LADDER_HF_PREFIX})",
        )
    except EntryNotFoundError:
        # Genuinely not-yet-uploaded — the ONLY case that legitimately means
        # "absent". Everything else re-raises.
        return False
    except RepositoryNotFoundError:
        # Deliberately NOT absence: a missing or inaccessible data repo is a
        # config / token-scope fault. Reading it as "manifest not uploaded"
        # would trigger a full rebuild + upload against the wrong target.
        # Matches the policy in the fits driver's reliability-ceiling probe.
        raise
    except HfHubHTTPError as exc:
        # NOTE (issue-1491): this was `except Exception: return False`, so a
        # transient 429/5xx or an auth fault read as "prefix absent" and
        # silently triggered a full rebuild + in-place re-upload of the
        # manifest — i.e. a network blip could REPLACE the pinned contexts
        # mid-experiment. A transport fault is not evidence of absence.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return False
        raise
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

    from explore_persona_space.orchestrate import hub  # type: ignore

    def _fetch(model_id: str, filename: str) -> str:
        """Download one tokenizer file, retrying the transient class.

        Bare hf_hub_download here meant a single 429/5xx during the 6-model
        gate crashed Phase 0 outright (the gate runs before any capture, so
        the whole run dies on a blip). retry_transient re-raises 404s
        unchanged, so a genuinely-missing file still fails loud.
        """
        return hub.retry_transient(
            lambda: hf_hub_download(repo_id=model_id, filename=filename),
            what=f"hf_hub_download({model_id}:{filename})",
        )

    per_model = []
    templates: set[str] = set()
    vocab_hashes: set[str] = set()
    merges_hashes: set[str] = set()
    tokenizer_json_hashes: set[str] = set()
    tokenizer_config_hashes: dict[str, str] = {}
    for model_id in QWEN_LADDER:
        cfg_local = _fetch(model_id, "tokenizer_config.json")
        with open(cfg_local, "rb") as fh:
            cfg_bytes = fh.read()
        cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
        tokenizer_config_hashes[model_id] = cfg_sha
        cfg = json.loads(cfg_bytes)
        chat_template = cfg.get("chat_template")
        assert chat_template, f"{model_id}: missing chat_template"
        templates.add(chat_template)
        vocab_local = _fetch(model_id, "vocab.json")
        with open(vocab_local, "rb") as fh:
            vocab_bytes = fh.read()
        vocab_hash = hashlib.sha256(vocab_bytes).hexdigest()
        vocab_hashes.add(vocab_hash)
        # BPE MERGE TABLE — load-bearing, and previously unchecked.
        #
        # NOTE (issue-1491): the gate hashed chat_template + vocab.json only.
        # Identical vocabularies with DIFFERENT merge tables segment the same
        # string differently, so a merge-table divergence across the ladder
        # would silently invalidate both things this gate exists to justify:
        # the shared over-length token budget, and the premise that one
        # tokenizer-derived context filter is valid for all six scales.
        merges_local = _fetch(model_id, "merges.txt")
        with open(merges_local, "rb") as fh:
            merges_bytes = fh.read()
        merges_hash = hashlib.sha256(merges_bytes).hexdigest()
        merges_hashes.add(merges_hash)
        # FAST-TOKENIZER FILE — the one AutoTokenizer actually loads.
        #
        # vocab.json + merges.txt are the SLOW-tokenizer inputs. `_make_token_len_fn`
        # builds its length function via AutoTokenizer, which for Qwen2 resolves the
        # fast tokenizer from tokenizer.json — so a tokenizer.json divergence
        # (added tokens, normalizer, pre-tokenizer settings) changes real
        # segmentation while passing a gate that only hashed the slow files. Same
        # class as the merges.txt gap this gate already had to close.
        # Probed live across all six ladder repos before asserting: a single
        # shared hash (c0382117ea329cdf...), so the assert covers what it claims
        # without false-firing.
        tokjson_local = _fetch(model_id, "tokenizer.json")
        with open(tokjson_local, "rb") as fh:
            tokjson_bytes = fh.read()
        tokjson_hash = hashlib.sha256(tokjson_bytes).hexdigest()
        tokenizer_json_hashes.add(tokjson_hash)
        per_model.append(
            {
                "model_id": model_id,
                "tokenizer_config_sha256": cfg_sha,
                "vocab_sha256": vocab_hash,
                "merges_sha256": merges_hash,
                "tokenizer_json_sha256": tokjson_hash,
            }
        )
    assert len(templates) == 1, (
        f"chat_template mismatch across ladder — {len(templates)} distinct templates"
    )
    assert len(vocab_hashes) == 1, (
        f"vocab hash mismatch across ladder — {len(vocab_hashes)} distinct vocabs"
    )
    assert len(merges_hashes) == 1, (
        f"BPE merge-table mismatch across ladder — {len(merges_hashes)} distinct "
        "merges.txt. Identical vocabs with different merge tables segment text "
        "differently, so neither the shared over-length token budget nor the "
        "single-tokenizer context filter is valid across these scales."
    )
    assert len(tokenizer_json_hashes) == 1, (
        f"fast-tokenizer mismatch across ladder — {len(tokenizer_json_hashes)} distinct "
        "tokenizer.json. This is the file AutoTokenizer actually loads, so a "
        "divergence here changes real segmentation even when the slow-tokenizer "
        "files (vocab.json + merges.txt) match."
    )
    chat_template = templates.pop()
    return {
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "vocab_sha256": next(iter(vocab_hashes)),
        "merges_sha256": next(iter(merges_hashes)),
        "tokenizer_json_sha256": next(iter(tokenizer_json_hashes)),
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
    # Deferred import: this module loads .env inside _load_env() rather than at
    # module top, so a module-top numpy would import BEFORE the shared-VM thread
    # caps (#847) bind. numpy is used only here, so deferring keeps the module
    # heavy-import-free. Pinned by tests/test_shared_vm_thread_caps.py
    # (test_no_new_torch_before_dotenv_vm_entrypoints).
    import numpy as np

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
    # The parent's documented recovery path (issue779_ffc_n1m_generate_capture
    # L186-214): applies the ORIGINAL round's fixed_split(5000, 3600, 400, 1000,
    # 42) val/test indices to round1 and returns the 1,400 prompts as
    # list(val) + list(test) — so [:400] is val_400 and [400:] is test_1000
    # (order-equivalence pinned by tests/test_issue1491_manifest_sha_domains.py
    # ::test_valtest_parent_return_value_order). The pre-fix code discarded this
    # return value and re-derived the same slices by hand.
    valtest = _valtest_prompts_from_round1(round1, check_ctx0=not smoke)
    assert len(valtest) == N_VAL_400 + N_TEST_1000, len(valtest)
    val_prompts = valtest[:N_VAL_400]
    test_prompts = valtest[N_VAL_400:]
    return round1, val_prompts, test_prompts


def _assert_pinned_membership(
    round1: list[str], val_prompts: list[str], test_prompts: list[str]
) -> None:
    """Three-domain membership check (port of #1776's fix, 04ce114b8fb2).

    Each pin is compared within its OWN domain (the P0 crash compared a
    prompt digest against index-array digests — see the constants block):

      1. round-1 prompt MEMBERSHIP: the parent hasher ``_sha_ids_or_prompts``
         (== ``N10._sha_prompts``, ``b"\\x00"``-separated) over the 5,000
         re-streamed round-1 prompts equals the frozen #779 n1m
         sampling-manifest ``used_shas.round1`` — the real stream-drift guard;
      2. split identity: the INDEX pins equal the shas recomputed from the
         committed #779 ``fair_comparison.json`` split params
         (``N50F._pinned_original_shas``) — passes by construction unless the
         split recipe drifts;
      3. composition: the derived val/test PROMPT digests equal the frozen
         prompt-domain pins.
    """
    got_r1 = _sha_ids_or_prompts(round1)
    assert got_r1 == ROUND1_PROMPT_SHA, (
        f"round-1 prompt-membership drift: {got_r1} != frozen {ROUND1_PROMPT_SHA} "
        "(#779 n1m sampling_manifest used_shas.round1) — the LMSYS stream changed; "
        "the pinned val-400/test-1000 cannot be recovered from a re-stream"
    )
    pinned = N50F._pinned_original_shas(N50F.DEFAULT_ORIG_DIR)
    assert pinned["val_sha256"] == VAL_400_INDEX_SHA, (
        f"val-400 INDEX-sha pin drifted from the #779 artifact: {pinned} != "
        f"{VAL_400_INDEX_SHA} (index-array domain, F._sha_ids)"
    )
    assert pinned["test_sha256"] == TEST_1000_INDEX_SHA, (
        f"test-1000 INDEX-sha pin drifted from the #779 artifact: {pinned} != "
        f"{TEST_1000_INDEX_SHA} (index-array domain, F._sha_ids)"
    )
    got_val = _sha_ids_or_prompts(val_prompts)
    got_test = _sha_ids_or_prompts(test_prompts)
    assert got_val == VAL_400_PROMPT_SHA, (
        f"pinned val-400 PROMPT-digest drift: {got_val} != {VAL_400_PROMPT_SHA}"
    )
    assert got_test == TEST_1000_PROMPT_SHA, (
        f"pinned test-1000 PROMPT-digest drift: {got_test} != {TEST_1000_PROMPT_SHA}"
    )
    print(
        "[ladder-manifest] pinned-membership confirmed: round1 prompt sha + "
        "index pins + val/test prompt digests all match (three-domain check)",
        flush=True,
    )


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
        round1, val_prompts, test_prompts = _derive_valtest_prompts(
            smoke=True, smoke_stream=smoke_stream
        )
    else:
        round1, val_prompts, test_prompts = _derive_valtest_prompts(smoke=False)

    assert len(val_prompts) == N_VAL_400, f"val_400 size wrong: {len(val_prompts)}"
    assert len(test_prompts) == N_TEST_1000, f"test_1000 size wrong: {len(test_prompts)}"
    # Three-domain membership check (production only; smoke uses synthetic streams).
    if not smoke:
        _assert_pinned_membership(round1, val_prompts, test_prompts)
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
        # Carry ladder_local_id like every other split — a consumer keying on it
        # (the capture driver's ci mapping does) would KeyError on tierB alone.
        # `i` IS the correct local id here: the skip-mask join directly below
        # matches this same enumerate index against train's ladder_local_id set.
        {**r, "split": "tierB_3600", "ladder_local_id": i}
        for i, r in enumerate(tierb_3600)
        if i in train_kept_ids
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
                "index_sha256": VAL_400_INDEX_SHA,
                "prompt_sha256": None if smoke else VAL_400_PROMPT_SHA,
            },
            "test_1000": {
                "kept": len(test_kept),
                "skipped": len(test_skip),
                "declared": N_TEST_1000,
                "index_sha256": TEST_1000_INDEX_SHA,
                "prompt_sha256": None if smoke else TEST_1000_PROMPT_SHA,
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
        # The three-domain membership pins the production build verified
        # (_assert_pinned_membership; None in smoke — synthetic streams).
        # Mirrors issue1776_contexts.py's split_pins block (04ce114b8fb2).
        "split_pins": None
        if smoke
        else {
            "val_400_index_sha": VAL_400_INDEX_SHA,
            "test_1000_index_sha": TEST_1000_INDEX_SHA,
            "round1_prompt_sha": ROUND1_PROMPT_SHA,
            "val_400_prompt_sha": VAL_400_PROMPT_SHA,
            "test_1000_prompt_sha": TEST_1000_PROMPT_SHA,
        },
        "output_files": SPLIT_FILES,
    }

    # Per-split CONTENT shas — the cross-scale-identity guard.
    #
    # NOTE (issue-1491): only val_400 / test_1000 carried a sha (the pinned
    # parent values). train_25k / wc_test_1k / tierB_3600 recorded COUNTS only,
    # so a rebuild producing DIFFERENT content at the SAME counts was
    # undetectable. That is the cross-scale-mismatch scenario this whole design
    # rests on avoiding: numpy does not guarantee Generator.choice stream
    # stability across versions, so a rebuild under a different numpy silently
    # re-draws the training contexts — and any scale already captured against
    # the previous draw becomes incomparable, with no signal anywhere. Hash
    # exactly the bytes that get uploaded.
    for split_name, fname in SPLIT_FILES.items():
        fpath = out_dir / fname
        if fpath.exists():
            meta["splits"].setdefault(split_name, {})["content_sha256"] = hashlib.sha256(
                fpath.read_bytes()
            ).hexdigest()

    (out_dir / MANIFEST_META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _remote_manifest_meta() -> dict | None:
    """Fetch the ALREADY-UPLOADED ladder meta.json, or None if absent.

    Transport faults re-raise (never read as absent) — same discipline as
    `_verify_ladder_manifest_present`.
    """
    from huggingface_hub import hf_hub_download  # type: ignore
    from huggingface_hub.errors import (  # type: ignore
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    from explore_persona_space.orchestrate import hub  # type: ignore

    try:
        # Retry the transient class before concluding absence — a blip here
        # would otherwise read as "no published manifest" and let the drift
        # refusal be skipped entirely. 404s re-raise unchanged.
        local = hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=LADDER_HF_REPO,
                filename=f"{LADDER_HF_PREFIX}/{MANIFEST_META_NAME}",
                repo_type="dataset",
            ),
            what=f"hf_hub_download({MANIFEST_META_NAME})",
        )
    except LocalEntryNotFoundError:
        # MUST precede the EntryNotFoundError branch — it is a SUBCLASS of it.
        # LocalEntryNotFoundError is the transient/offline class (no cached copy
        # AND the Hub was unreachable); retry_transient re-raises it once the
        # budget exhausts. Falling through to `return None` would report a
        # sustained transport outage as "nothing published", which SKIPS the
        # drift refusal below — the precise fail-open that refusal exists to
        # prevent (an in-place re-upload replacing the pinned contexts every
        # scale is captured against).
        raise
    except EntryNotFoundError:
        return None
    except RepositoryNotFoundError:
        # Not absence — a config / token-scope fault. Same policy as
        # _verify_ladder_manifest_present and the fits ceiling probe.
        raise
    except HfHubHTTPError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            return None
        raise
    with open(local, encoding="utf-8") as fh:
        return json.load(fh)


def assert_no_silent_content_drift(new_meta: dict) -> None:
    """Refuse an in-place re-upload that would CHANGE any split's content.

    A `--force` rebuild is legitimate for re-uploading identical bytes (a
    partial upload, a corrupted remote file). It is NOT legitimate for
    replacing the pinned contexts of an experiment whose earlier scales have
    already been captured: those captures silently become incomparable with the
    later ones, and the cross-scale comparison — the entire point of the
    ladder — is invalidated with no signal.

    So: identical content re-uploads freely; CHANGED content must go to a new
    `recipe_version` (a version bump), never over the existing prefix.
    """
    remote = _remote_manifest_meta()
    if remote is None:
        return  # nothing published yet — nothing to drift from
    drifted = []
    for split_name, new_split in new_meta.get("splits", {}).items():
        new_sha = new_split.get("content_sha256")
        old_sha = remote.get("splits", {}).get(split_name, {}).get("content_sha256")
        if new_sha and old_sha and new_sha != old_sha:
            drifted.append((split_name, old_sha[:12], new_sha[:12]))
    if drifted:
        detail = "; ".join(f"{s}: {o} -> {n}" for s, o, n in drifted)
        raise RuntimeError(
            "REFUSING in-place re-upload — split content changed vs the published "
            f"manifest ({detail}). Any scale already captured against the published "
            "manifest would become incomparable with scales captured against this "
            "one, silently invalidating the cross-scale comparison. Publish under a "
            "new recipe_version / prefix instead of overwriting. "
            f"(published recipe_version={remote.get('recipe_version')!r}, "
            f"new={new_meta.get('recipe_version')!r})"
        )
    if remote.get("recipe_version") and not any(
        remote.get("splits", {}).get(s, {}).get("content_sha256")
        for s in new_meta.get("splits", {})
    ):
        # Published BEFORE content shas existed — cannot prove identity.
        print(
            "[manifest] WARNING: published manifest predates content shas; "
            "identity with the new build is UNPROVEN. Verify before reusing "
            "captures made against it.",
            flush=True,
        )


def upload_manifest(local_dir: Path) -> None:
    """Upload the manifest tree as ONE ``upload_folder`` commit."""
    api = _hf_api()
    logger.info(
        "uploading %s -> %s:%s",
        local_dir,
        LADDER_HF_REPO,
        LADDER_HF_PREFIX,
    )
    from explore_persona_space.orchestrate import hub  # type: ignore

    # Dir-filecount guard BEFORE the upload and deliberately OUTSIDE the retry
    # wrapper: the Hub rejects >10k files in one repo dir with a NON-retriable
    # BadRequestError fired after all bytes are staged (#658), and a guard raise
    # is deterministic — retrying it would burn the retry budget for nothing.
    # This manifest commits only the split files + meta, far under the limit, so
    # the guard is a cheap invariant rather than an expected trigger.
    hub.assert_hub_dir_filecounts(
        local_dir,
        LADDER_HF_PREFIX,
        allow_patterns=list(SPLIT_FILES.values()) + [MANIFEST_META_NAME],
    )

    # The manifest upload is the single durability point for the pinned
    # contexts every scale is captured against — a transient 429/5xx here
    # must retry, not abort Phase 0 after the whole build already ran.
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=LADDER_HF_PREFIX,
            repo_id=LADDER_HF_REPO,
            repo_type="dataset",
            commit_message=(
                f"issue1491 ladder manifest v1 (train_25k+val+test+wc+tierB; "
                f"budget={OVERLENGTH_BUDGET})"
            ),
            allow_patterns=list(SPLIT_FILES.values()) + [MANIFEST_META_NAME],
        ),
        what=f"upload_folder({LADDER_HF_PREFIX})",
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
    # Fail BEFORE the upload: a rebuild whose content differs from the published
    # manifest must never silently replace it (see assert_no_silent_content_drift).
    assert_no_silent_content_drift(meta)
    upload_manifest(args.out_dir)
    print(f"UPLOAD OK — wrote and uploaded {LADDER_HF_REPO}:{LADDER_HF_PREFIX}")
    print(json.dumps(meta["splits"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
