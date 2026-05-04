"""Light unit tests for `scripts/issue224_attention_analysis.py`.

These do NOT require a GPU or model load. They exercise the pure-Python
helpers (region boundary computation, marker timestep selection, segmentation
B partition exhaustiveness) on small toy inputs. The remaining integration
testing (forward pass + hooks + attention shape) lives on the pod.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Importable handle to the script-as-module (it is at scripts/, not in a
# package). Use importlib so we don't pollute the test runner with hooks.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "issue224_attention_analysis.py"
spec = importlib.util.spec_from_file_location("issue224_attn", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["issue224_attn"] = mod
spec.loader.exec_module(mod)


# ── Toy tokenizer for region_boundaries / marker selection / segmentation ────


class _ToyTokenizer:
    """Mimics the tokenizer surface that the script touches.

    Encodes by mapping each character to its Unicode codepoint, treating each
    char as one token. Special tokens get reserved ids in [10000, 10100). The
    chat template is a deterministic, simple wrap.
    """

    def __init__(self) -> None:
        self.eos_token_id = 10001
        self.unk_token_id = 10099
        # Reserved structural specials with stable ids.
        self.specials: dict[str, int] = {
            "<|im_start|>": 10000,
            "<|im_end|>": 10001,
            "<|endoftext|>": 10002,
            # `\n` and `\n\n` are real ASCII codepoints — keep `\n\n` at 271
            # to match the script's NEWLINE2 constant.
        }
        # Every "[Z" / "LT" / "]" must encode to ZLT_FIRST/27404/60.
        self._zlt_alias = {"[Z": 85113, "LT": 27404, "]": 60}

    # Standard tokenizer surface ─────────────────────────────────────────────

    def get_vocab(self) -> dict[str, int]:
        # Minimal set: ASCII printable plus specials and `\n\n`.
        v: dict[str, int] = {chr(c): c for c in range(32, 127)}
        v["\n"] = 10  # ASCII LF
        v["\n\n"] = 271  # NEWLINE2
        v.update(self.specials)
        return v

    def convert_tokens_to_ids(self, s: str) -> int:
        return self.specials.get(s, self.unk_token_id)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        rev = {v: k for k, v in self.specials.items()}
        for tid in ids:
            if tid == 271:
                out.append("\n\n")
            elif tid in rev:
                out.append("" if skip_special_tokens else rev[tid])
            elif 32 <= tid <= 126:
                out.append(chr(tid))
            elif tid == 10:
                out.append("\n")
            elif tid in (85113, 27404, 60):
                # Decode marker BPE pieces consistently.
                rev_zlt = {85113: "[Z", 27404: "LT", 60: "]"}
                out.append(rev_zlt[tid])
            else:
                out.append(f"<{tid}>")
        return "".join(out)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Greedy multi-char alias matching for `[Z` / `LT` / `]` / `\n\n` /
        # specials. Fallback: per-character.
        i = 0
        out: list[int] = []
        keys = sorted(self._zlt_alias.keys(), key=len, reverse=True)
        special_keys = sorted(self.specials.keys(), key=len, reverse=True)
        while i < len(text):
            matched = False
            for s in special_keys:
                if text.startswith(s, i):
                    out.append(self.specials[s])
                    i += len(s)
                    matched = True
                    break
            if matched:
                continue
            if text.startswith("\n\n", i):
                out.append(271)
                i += 2
                continue
            for s in keys:
                if text.startswith(s, i):
                    out.append(self._zlt_alias[s])
                    i += len(s)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            out.append(ord(ch))
            i += 1
        return out

    # Chat template ──────────────────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if tokenize:
            raise NotImplementedError("only str output is exercised in these tests")
        return text


# ── region_boundaries ───────────────────────────────────────────────────────


def test_region_boundaries_sums_to_full_prompt_length() -> None:
    tok = _ToyTokenizer()
    sys_prompt = "You are a librarian."
    user_q = "What is photosynthesis?"
    n_sys, n_user, n_hdr = mod.region_boundaries(tok, sys_prompt, user_q)
    full_text = tok.apply_chat_template(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_q},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_len = len(tok.encode(full_text, add_special_tokens=False))
    assert n_sys + n_user + n_hdr == full_len, (
        f"region_boundaries should sum to prompt length: "
        f"n_sys={n_sys} n_user={n_user} n_hdr={n_hdr} full_len={full_len}"
    )
    assert n_sys > 0 and n_user > 0 and n_hdr > 0


def test_region_boundaries_increases_with_user_length() -> None:
    tok = _ToyTokenizer()
    sys_prompt = "p"
    short = "q"
    longer = "q" * 20
    _, n_user_short, _ = mod.region_boundaries(tok, sys_prompt, short)
    _, n_user_long, _ = mod.region_boundaries(tok, sys_prompt, longer)
    assert n_user_long > n_user_short


# ── marker timestep selection ───────────────────────────────────────────────


def test_marker_first_zbracket_selected() -> None:
    """The first `[Z` (id 85113) in the assistant span is the marker timestep."""
    # Synthetic: prompt of 5 tokens, asst content with answer + [ZLT] at the end.
    prompt = [10000, 99, 99, 99, 10001]  # <im_start>, junk, <im_end>
    asst = [70, 71, 72, 73, 271, mod.ZLT_FIRST, 27404, 60, 10001]
    full = prompt + asst
    # Detect first `[Z` position
    assert mod.ZLT_FIRST in full
    pos = full.index(mod.ZLT_FIRST)
    assert pos == len(prompt) + 5  # after 4 content + \n\n
    assert mod._zlt_in_token_ids(asst) is True
    # Position 60 (`]`) is later in full; make sure it isn't picked as marker
    assert pos < full.index(60)


def test_marker_skipped_when_absent() -> None:
    asst = [70, 71, 72, 271, 10001]  # answer, then \n\n, then <im_end>; no [ZLT]
    assert mod._zlt_in_token_ids(asst) is False


# ── segmentation B exhaustive partition ─────────────────────────────────────


def test_segmentation_b_partitions_to_unity() -> None:
    """For toy attention rows (sum-to-1 by construction), seg-B regions plus
    the `specials` bucket exhaust the keys [0, t]. Partition sum-to-1 must
    match within tolerance.
    """
    rng = np.random.default_rng(0)
    # Build toy positional layout: n_sys=4, n_user=3, n_hdr=2 (header) + 5 content
    n_sys, n_user, n_hdr = 4, 3, 2
    asst_start = n_sys + n_user + n_hdr
    asst_content_len = 5
    full_ids: list[int] = []
    # System block: <im_start> + 2 content + <im_end>
    full_ids.extend([10000, 65, 66, 10001])  # 4 tokens
    # User block: 3 tokens (no specials, simple content for the test)
    full_ids.extend([67, 68, 69])
    # Asst header: <im_start>assistant
    full_ids.extend([10000, 70])  # 2 tokens — both flagged structural here
    # Asst content: 4 content tokens then \n\n (structural)
    full_ids.extend([71, 72, 73, 74, 271])
    # `t` is the last position
    t = len(full_ids) - 1
    # Synthesize captures: each layer's row at position t is a softmax over
    # k=0..t. Use softmax of random logits to guarantee row-sum-1 numerically.
    n_layers = mod.N_LAYERS
    n_heads = mod.N_HEADS
    captures = []
    for _L in range(n_layers):
        # Shape (1, n_heads, q_len=t+1, k_len=t+1) but we only access row t.
        # Build a dense matrix sized (n_heads, t+1, t+1) and softmax along axis=-1
        logits = rng.standard_normal((n_heads, t + 1, t + 1)).astype(np.float32)
        # Causal mask: zero out k > q (won't be touched at row t since k <= t).
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        attn = e / e.sum(axis=-1, keepdims=True)
        # Mock a torch-tensor-like API used by the script.
        captures.append(_ToyTensor(attn))
    # Build structural set: includes 10000, 10001, 271 (and ASCII LF=10 if any).
    structural = {10000, 10001, 271}

    out = mod._attn_at_t_to_regions(captures, t, n_sys, n_user, n_hdr, structural, full_ids)
    # Per layer + per head, the four-region B partition must sum to 1
    for L in range(n_layers):
        sum_b = (
            np.array(out["system_B"])[L]
            + np.array(out["user_B"])[L]
            + np.array(out["asst_B"])[L]
            + np.array(out["specials"])[L]
        )
        assert np.allclose(sum_b, 1.0, atol=mod.PARTITION_SUM_TOL), (
            f"layer={L} seg-B partition sum != 1 (max dev {np.max(np.abs(sum_b - 1)):.3e})"
        )
    # Also check segmentation A
    for L in range(n_layers):
        sum_a = (
            np.array(out["system_A"])[L]
            + np.array(out["user_A"])[L]
            + np.array(out["asst_header_A"])[L]
            + np.array(out["asst_A"])[L]
        )
        assert np.allclose(sum_a, 1.0, atol=mod.PARTITION_SUM_TOL), (
            f"layer={L} seg-A partition sum != 1 (max dev {np.max(np.abs(sum_a - 1)):.3e})"
        )
    # Quick reference: asst_content_len must include the structural \n\n we
    # appended; ensure the content-only asst_B is nonzero (one of 71..74)
    assert np.any(np.array(out["asst_B"]) > 0)
    # silence "unused" linter on context vars
    assert asst_start == 9 and asst_content_len == 5


class _ToyTensor:
    """Minimal stand-in for a torch.Tensor with the few attributes the script
    accesses on captured attention: ``[0, :, t, :src_len].float().cpu().numpy()``.
    """

    def __init__(self, arr: np.ndarray) -> None:
        # Add the leading batch dim to match (1, n_heads, q, k)
        self._arr = arr[np.newaxis, ...]

    def __getitem__(self, key):
        return _ToyTensor2(self._arr[key])


class _ToyTensor2:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


# ── _zlt_in_text vs _zlt_in_token_ids consistency ──────────────────────────


def test_substring_and_tokenid_detectors_consistent_on_basic_cases() -> None:
    """Token-id-based detector and substring-based detector should agree
    on all completion strings the toy tokenizer can round-trip.
    """
    tok = _ToyTokenizer()
    cases = [
        "hello world",
        "the answer is 42 [ZLT]",
        "[ZLT] right at the start",
        "no marker here",
        "ZLT without brackets",  # should not match
    ]
    for text in cases:
        ids = tok.encode(text, add_special_tokens=False)
        assert mod._zlt_in_text(text) == mod._zlt_in_token_ids(ids), (
            f"detector disagreement on text={text!r}"
        )


# ── structural-set composition (BLOCKER 2 fix verification) ────────────────


def test_structural_set_excludes_spaces() -> None:
    """Code-review v1 BLOCKER 2 verification: structural_token_ids must
    contain only chat-template specials and the literal newline tokens
    `"\\n"` / `"\\n\\n"` — NOT pure-space BPEs (which would over-strip
    content under segmentation B and inflate the `specials` bucket).

    Loads the real Qwen-2.5-7B-Instruct tokenizer (CPU-only, no GPU
    required). Skips if HF Hub is unreachable.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        pytest.skip(f"transformers not importable in test env: {exc}")

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"could not download Qwen tokenizer (network?): {exc}")

    structural = mod.structural_token_ids(tok)
    assert isinstance(structural, set) and len(structural) > 0

    # ── Pure-space tokens MUST NOT be structural ────────────────────────
    space_strings = (" ", "  ", " word", "the ")
    for s in space_strings:
        ids = tok.encode(s, add_special_tokens=False)
        for tid in ids:
            assert tid not in structural, (
                f"pure-space token id {tid} (decodes to "
                f"{tok.decode([tid])!r} from input {s!r}) "
                "must NOT be structural — segmentation B would over-strip it"
            )

    # Sample content tokens via a short phrase: none of these should be
    # in the structural set.
    content_ids = tok.encode("the world ", add_special_tokens=False)
    for tid in content_ids:
        decoded = tok.decode([tid])
        if decoded in ("\n", "\n\n"):
            continue  # if a content phrase happens to encode a newline, that's fine
        assert tid not in structural, (
            f"content token id {tid} (decodes to {decoded!r}) must not be structural"
        )

    # ── Newline tokens MUST be structural ───────────────────────────────
    nl_ids = tok.encode("\n", add_special_tokens=False)
    assert nl_ids, "expected `\\n` to encode to at least one token"
    # Most BPE-tokenizers encode "\n" as a single id; we assert all ids in
    # the encoding (typically 1) are in the structural set.
    for tid in nl_ids:
        assert tid in structural, (
            f"`\\n` token id {tid} (decoded={tok.decode([tid])!r}) must be structural"
        )

    nl2_ids = tok.encode("\n\n", add_special_tokens=False)
    assert nl2_ids, "expected `\\n\\n` to encode to at least one token"
    for tid in nl2_ids:
        assert tid in structural, (
            f"`\\n\\n` token id {tid} (decoded={tok.decode([tid])!r}) must be structural"
        )

    # ── Chat-template special tokens MUST be structural when in vocab ───
    for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(special)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            assert tid in structural, (
                f"chat-template special {special!r} (id={tid}) must be structural"
            )


# ── pytest mainline ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
