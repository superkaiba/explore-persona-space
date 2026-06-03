"""Phase 0 — preflight for #474 on-policy divergence-to-transfer with
localization restored.

Issue #474 plan v3 §4.1. Forked from ``scripts/i460_phase0_preflight.py``
with the following additions on top of #460's checks (1-5):

  6. **DYNAMIC slot-layout asserts (M1 fix)** — verified Qwen-2.5-7B-Instruct
     this session:

     POSITIVE tail layout ``...Answer.[L-4] ※[L-3] <|im_end|>[L-2] \\n[L-1]``
     NEGATIVE tail layout ``...Answer.[L-3] <|im_end|>[L-2] \\n[L-1]``

     The POSITIVE marker label slot ``pos_ids[-3]`` (loss target for positives,
     conditioning context = "...Answer.") and the NEGATIVE first-<|im_end|>
     label slot ``neg_ids[-2]`` (loss target for A_loc negatives, same
     conditioning context = "...Answer.") share the SAME context. Suppressing
     log P(※) at the negative's loss slot under softmax competition pushes
     log P(※) DOWN at the positive's MEASUREMENT slot.

  7. **Frozen R artifact** presence check (#460 produces ``R_train.json``
     / ``R_test.json`` on HF data repo ``superkaiba1/explore-persona-space-data``
     under ``issue460_marker_at_end/on_policy_R/``; A_pos and A_loc SHARE this
     same R artifact).

  8. **posix_fallocate ~10 GB probe on /workspace** (MooseFS quota check).

  9. **vLLM full-vocab probe** (sets fallback flag for Phase 4 KL if
     ``prompt_logprobs=152064`` is rejected). Gated by ``--skip-vllm-probe``
     for CPU-only smoke / preflight.

CLI:
    uv run python scripts/i474_phase0_preflight.py
    uv run python scripts/i474_phase0_preflight.py --skip-vllm-probe --skip-disk-probe
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (
    HF_DATA_REPO,
    assert_disjoint_q_train_q_test,
    load_class_d_rewrites,
    load_q_test_extended_50,
    load_q_train_answers,
)

logger = logging.getLogger("i474.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
D_MATRIX_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
G_MATRIX_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
OUT_DIR = Path("eval_results/issue_474")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"

HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"

# Disk-probe defaults: ~10 GB matches plan v3 §4.1 + gotchas.md MooseFS quota.
DEFAULT_DISK_PROBE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
DEFAULT_DISK_PROBE_DIR = Path("/workspace")


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_slot_layout(tokenizer) -> dict:
    """Verify the Qwen-2.5 chat-template tail layout dynamically.

    Returns a dict with the resolved IM_END_ID / NEWLINE_ID for the
    preflight payload. Fails loud on any drift.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise AssertionError(
            f"<|im_end|> not resolved by tokenizer (got {im_end_id}). "
            "Tokenizer drift — Qwen-2.5-7B-Instruct expected."
        )
    if im_end_id != 151645:
        raise AssertionError(
            f"<|im_end|> id drift: got {im_end_id}, expected 151645 "
            "(Qwen-2.5 tokenizer). Refusing to launch."
        )

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_ids:
        raise AssertionError("tokenizer.encode('\\n') returned empty.")
    newline_id = newline_ids[-1]
    if newline_id != 198:
        raise AssertionError(
            f"Newline id drift: got {newline_id}, expected 198 (Qwen-2.5 tokenizer)."
        )

    # Synthetic positive: T(q) + Answer. + " ※"
    pos_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": f"Answer.{MARKER_TEXT}"},
    ]
    pos_text = tokenizer.apply_chat_template(pos_msgs, tokenize=False, add_generation_prompt=False)
    pos_ids = tokenizer.encode(pos_text, add_special_tokens=False)

    # Synthetic negative: T_j(q) + Answer. (no marker)
    neg_msgs = [
        {"role": "system", "content": "You are a swashbuckling pirate captain."},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "Answer."},
    ]
    neg_text = tokenizer.apply_chat_template(neg_msgs, tokenize=False, add_generation_prompt=False)
    neg_ids = tokenizer.encode(neg_text, add_special_tokens=False)

    # POSITIVE tail asserts.
    if pos_ids[-3] != MARKER_ID:
        raise AssertionError(
            f"positive marker label slot drift: pos_ids[-3]={pos_ids[-3]} "
            f"(expected MARKER_ID {MARKER_ID}). pos_ids[-5:]={pos_ids[-5:]}"
        )
    if pos_ids[-2] != im_end_id:
        raise AssertionError(
            f"positive <|im_end|> slot drift: pos_ids[-2]={pos_ids[-2]} "
            f"(expected IM_END_ID {im_end_id}). pos_ids[-5:]={pos_ids[-5:]}"
        )
    if pos_ids[-1] != newline_id:
        raise AssertionError(
            f"positive trailing newline drift: pos_ids[-1]={pos_ids[-1]} "
            f"(expected NEWLINE {newline_id}). pos_ids[-5:]={pos_ids[-5:]}"
        )

    # NEGATIVE tail asserts.
    if neg_ids[-2] != im_end_id:
        raise AssertionError(
            f"negative post-response slot drift: neg_ids[-2]={neg_ids[-2]} "
            f"(expected IM_END_ID {im_end_id}). neg_ids[-5:]={neg_ids[-5:]}"
        )
    if neg_ids[-1] != newline_id:
        raise AssertionError(
            f"negative trailing newline drift: neg_ids[-1]={neg_ids[-1]} "
            f"(expected NEWLINE {newline_id}). neg_ids[-5:]={neg_ids[-5:]}"
        )
    if MARKER_ID in neg_ids:
        raise AssertionError(
            f"negative row contains MARKER_ID {MARKER_ID} — would corrupt the "
            "no-marker contrastive negative. neg_ids[-5:]={neg_ids[-5:]}"
        )

    logger.info(
        "Slot layout OK: IM_END_ID=%d, MARKER_ID=%d, NEWLINE_ID=%d, pos tail=%s, neg tail=%s",
        im_end_id,
        MARKER_ID,
        newline_id,
        pos_ids[-5:],
        neg_ids[-5:],
    )
    return {
        "im_end_id": im_end_id,
        "newline_id": newline_id,
        "pos_tail_ids": pos_ids[-5:],
        "neg_tail_ids": neg_ids[-5:],
    }


def _check_frozen_r_on_hf() -> dict:
    """Verify the #460 frozen R artifact is reachable on HF data repo."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(
        HF_DATA_REPO,
        repo_type="dataset",
        revision="main",
    )
    required = [
        f"{HF_R_PATH_PREFIX}/R_train.json",
        f"{HF_R_PATH_PREFIX}/R_test.json",
    ]
    missing = [f for f in required if f not in files]
    if missing:
        raise FileNotFoundError(
            f"#460 frozen R missing on HF data repo {HF_DATA_REPO}: {missing}. "
            "A_pos and A_loc SHARE this R; cannot proceed."
        )
    logger.info("Frozen R OK on HF: %s", required)
    return {"hf_data_repo": HF_DATA_REPO, "r_paths": required}


def _disk_probe(target_dir: Path, n_bytes: int) -> dict:
    """posix_fallocate probe — catches the MooseFS per-pod EDQUOT quota."""
    target_dir.mkdir(parents=True, exist_ok=True)
    probe = target_dir / f".i474_disk_probe_{os.getpid()}"
    import contextlib

    fd = os.open(str(probe), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.posix_fallocate(fd, 0, n_bytes)
    except OSError as e:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)
        raise RuntimeError(
            f"posix_fallocate({n_bytes} bytes) FAILED on {target_dir}: "
            f"errno={e.errno} {e.strerror}. "
            "MooseFS quota likely - see CLAUDE.md gotchas (EDQUOT 122)."
        ) from e
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    with contextlib.suppress(OSError):
        os.unlink(probe)
    logger.info("Disk probe OK: %d bytes on %s", n_bytes, target_dir)
    return {"target_dir": str(target_dir), "bytes": n_bytes}


def _vllm_full_vocab_probe(vocab_size: int) -> dict:
    """vLLM full-vocab prompt_logprobs probe.

    Returns ``{"supports_full_vocab_prompt_logprobs": True/False, "vocab_size": ...}``.
    Phase 4 reads this to choose between full-vocab KL and the 10000+tail-mass
    fallback (`Source: feedback_route_b_kl_dv_swap.md`).
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        logger.warning("vLLM import failed (CPU-only env?): %s", e)
        return {
            "supports_full_vocab_prompt_logprobs": False,
            "vocab_size": vocab_size,
            "probe_skipped_reason": f"vllm import failed: {e}",
        }

    try:
        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.30,
            max_model_len=2048,
            enforce_eager=True,
        )
        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=vocab_size,
        )
        llm.generate(["Hello"], sampling_params=sp)
        supports = True
    except Exception as e:
        logger.warning("Full-vocab prompt_logprobs (%d) REJECTED by vLLM: %s", vocab_size, e)
        supports = False

    return {
        "supports_full_vocab_prompt_logprobs": supports,
        "vocab_size": vocab_size,
    }


def main(argv: list[str] | None = None) -> None:  # noqa: C901  preflight steps are sequential gates, refactor would reduce readability
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="Skip writing preflight.json.")
    ap.add_argument(
        "--skip-vllm-probe",
        action="store_true",
        help="Skip the vLLM full-vocab probe (CPU-only env / preflight-only).",
    )
    ap.add_argument(
        "--skip-disk-probe",
        action="store_true",
        help="Skip the /workspace posix_fallocate probe (no MooseFS / CPU-only env).",
    )
    ap.add_argument(
        "--skip-frozen-r-check",
        action="store_true",
        help="Skip the HF frozen-R presence check (no network).",
    )
    ap.add_argument(
        "--disk-probe-dir",
        type=Path,
        default=DEFAULT_DISK_PROBE_DIR,
        help="Override the disk-probe target directory (default /workspace).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # 1. Marker token id assert + slot-layout asserts (M1 fix).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(
            f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]. "
            "Refusing to launch with marker drift."
        )
    logger.info("Marker token id OK: %s -> %d", MARKER_TEXT, MARKER_ID)

    slot_layout = _check_slot_layout(tokenizer)

    # 2. CONDITIONS sanity.
    if len(CONDITIONS) != 16:
        raise AssertionError(f"Expected 16 active conditions, got {len(CONDITIONS)}.")
    logger.info("CONDITIONS = %d (A1..D5 minus dropped C2..C5)", len(CONDITIONS))

    # 3. D_matrix schema.
    if not D_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"#406 D_matrix.json missing at {D_MATRIX_PATH}. "
            "v3 reuses #406's predictor; the matrix must be on the branch."
        )
    d_payload = json.loads(D_MATRIX_PATH.read_text())
    if d_payload.get("n_conditions") != 16:
        raise AssertionError(
            f"D_matrix.json n_conditions = {d_payload.get('n_conditions')}, expected 16."
        )
    if d_payload["KL"]["A1"]["A1"] is not None:
        raise AssertionError(
            f"D_matrix.json KL[A1][A1] = {d_payload['KL']['A1']['A1']!r}, "
            "expected None (diagonal cells must be None)."
        )
    if (
        not isinstance(d_payload["KL"]["A1"]["B1"], (int, float))
        or d_payload["KL"]["A1"]["B1"] <= 0
    ):
        raise AssertionError(
            f"D_matrix.json KL[A1][B1] = {d_payload['KL']['A1']['B1']!r}, "
            "expected a positive float."
        )
    d_hash = _file_sha256(D_MATRIX_PATH)
    logger.info("D_matrix.json schema OK (sha256[:12]=%s)", d_hash[:12])

    # 4. G_matrix schema (#406 head-to-head DESCRIPTIVE comparison).
    if not G_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"#406 G_matrix.json missing at {G_MATRIX_PATH}. "
            "v3's DESCRIPTIVE #406 head-to-head reads from this file."
        )
    g_payload = json.loads(G_MATRIX_PATH.read_text())
    if g_payload.get("n_conditions") != 16:
        raise AssertionError(
            f"G_matrix.json n_conditions = {g_payload.get('n_conditions')}, expected 16."
        )
    if g_payload["G"]["A1"]["A1"] is None:
        raise AssertionError(
            "G_matrix.json G[A1][A1] is None — expected a {n_emit, n_total, rate} dict."
        )
    if not isinstance(g_payload["G"]["A1"]["B1"], dict) or "rate" not in g_payload["G"]["A1"]["B1"]:
        raise AssertionError(
            f"G_matrix.json G[A1][B1] = {g_payload['G']['A1']['B1']!r}, "
            "expected a {n_emit, n_total, rate} dict."
        )
    g_hash = _file_sha256(G_MATRIX_PATH)
    logger.info("G_matrix.json schema OK (sha256[:12]=%s)", g_hash[:12])

    # 5. Q_train / Q_test / Class-D loadable (with HF fallback).
    q_train = load_q_train_answers()
    q_test = load_q_test_extended_50()
    class_d = load_class_d_rewrites()
    assert_disjoint_q_train_q_test(list(q_train.keys()), q_test)
    missing_qs = [q for q in list(q_train.keys()) + q_test if q not in class_d]
    if missing_qs:
        raise AssertionError(
            f"Class-D rewrites missing for {len(missing_qs)} questions; first: {missing_qs[0]!r}"
        )
    logger.info(
        "Q_train=%d Q_test=%d class_d=%d (disjoint, full Q coverage)",
        len(q_train),
        len(q_test),
        len(class_d),
    )

    # 6/7/8/9 — #474 additions (gated).
    frozen_r_info = None
    if not args.skip_frozen_r_check:
        frozen_r_info = _check_frozen_r_on_hf()

    disk_info = None
    if not args.skip_disk_probe:
        disk_info = _disk_probe(args.disk_probe_dir, DEFAULT_DISK_PROBE_BYTES)

    vllm_info = None
    if not args.skip_vllm_probe:
        vllm_info = _vllm_full_vocab_probe(tokenizer.vocab_size)

    payload = {
        "schema_version": "i474_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "n_conditions": len(CONDITIONS),
        "condition_ids": [c.cid for c in CONDITIONS],
        "n_q_train": len(q_train),
        "n_q_test": len(q_test),
        "d_matrix_path": str(D_MATRIX_PATH),
        "d_matrix_sha256": d_hash,
        "g_matrix_path": str(G_MATRIX_PATH),
        "g_matrix_sha256": g_hash,
        "slot_layout": slot_layout,
        "frozen_r": frozen_r_info,
        "disk_probe": disk_info,
        "vllm_full_vocab_probe": vllm_info,
    }
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2))
        logger.info("Preflight OK -> %s", PREFLIGHT_PATH)
    else:
        logger.info("Preflight OK (dry-run; skipping write)")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
