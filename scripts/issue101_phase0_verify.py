#!/usr/bin/env python3
"""Issue #101 Phase 0: Verify Qwen chat template behavior for 4 conditions.

Confirms:
1. qwen_default and no_system_sanity produce IDENTICAL tokens
2. generic_assistant differs in system content
3. empty_system has <|im_start|>system\n<|im_end|> with no content

No GPU needed -- tokenizer only.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

# ── Conditions ───────────────────────────────────────────────────────────────

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""  # empty string -> empty system block

CONDITIONS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
    "no_system_sanity": None,  # None = omit system message entirely
}

TEST_QUESTION = "What causes earthquakes?"


def build_messages(persona_prompt, question):
    """Build chat messages with explicit empty-system support."""
    messages = []
    if persona_prompt is not None:
        messages.append({"role": "system", "content": persona_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def main():
    print("=" * 70)
    print("Issue #101 Phase 0: Chat Template Verification")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    results = {}
    for name, prompt in CONDITIONS.items():
        messages = build_messages(prompt, TEST_QUESTION)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        results[name] = {"text": text, "tokens": tokens, "n_tokens": len(tokens)}

        print(f"\n{'─' * 60}")
        print(f"Condition: {name}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Rendered text:\n{text}")
        print(f"Token count: {len(tokens)}")

    # ── Verification checks ──────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    all_pass = True

    # Check 1: qwen_default == no_system_sanity (identical tokens)
    qd_tokens = results["qwen_default"]["tokens"]
    ns_tokens = results["no_system_sanity"]["tokens"]
    check1 = qd_tokens == ns_tokens
    status = "PASS" if check1 else "FAIL"
    print(f"\n[{status}] Check 1: qwen_default == no_system_sanity tokens")
    if not check1:
        all_pass = False
        print(f"  qwen_default:     {len(qd_tokens)} tokens")
        print(f"  no_system_sanity: {len(ns_tokens)} tokens")
        # Show diff
        for i, (a, b) in enumerate(zip(qd_tokens, ns_tokens, strict=False)):
            if a != b:
                print(f"  First diff at position {i}: {a} vs {b}")
                break

    # Check 2: generic_assistant differs from qwen_default
    ga_tokens = results["generic_assistant"]["tokens"]
    check2 = ga_tokens != qd_tokens
    status = "PASS" if check2 else "FAIL"
    print(f"[{status}] Check 2: generic_assistant != qwen_default tokens")
    if not check2:
        all_pass = False
        print("  ERROR: These should differ!")

    # Check 3: empty_system has empty system block
    es_text = results["empty_system"]["text"]
    # The empty system should have the system tags with nothing between them
    has_empty_system = "<|im_start|>system\n<|im_end|>" in es_text
    status = "PASS" if has_empty_system else "FAIL"
    print(f"[{status}] Check 3: empty_system has empty system block")
    if not has_empty_system:
        all_pass = False
        print("  Looking for: <|im_start|>system\\n<|im_end|>")
        print(f"  Got: {es_text[:200]}")

    # Check 4: empty_system differs from both qwen_default and generic_assistant
    es_tokens = results["empty_system"]["tokens"]
    check4a = es_tokens != qd_tokens
    check4b = es_tokens != ga_tokens
    status = "PASS" if (check4a and check4b) else "FAIL"
    print(f"[{status}] Check 4: empty_system differs from both others")

    # Check 5: qwen_default text contains "Qwen" and "Alibaba"
    qd_text = results["qwen_default"]["text"]
    check5 = "Qwen" in qd_text and "Alibaba" in qd_text
    status = "PASS" if check5 else "FAIL"
    print(f"[{status}] Check 5: qwen_default text contains 'Qwen' and 'Alibaba'")

    # Summary
    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- investigate before proceeding")
        sys.exit(1)

    # Save results
    output_dir = Path("eval_results/issue101")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase0_verification.json"

    save_data = {}
    for name, data in results.items():
        save_data[name] = {
            "text": data["text"],
            "n_tokens": data["n_tokens"],
            "first_20_tokens": data["tokens"][:20],
        }
    save_data["checks"] = {
        "qwen_default_eq_no_system_sanity": qd_tokens == ns_tokens,
        "generic_assistant_ne_qwen_default": ga_tokens != qd_tokens,
        "empty_system_has_empty_block": has_empty_system,
        "empty_system_ne_qwen_default": es_tokens != qd_tokens,
        "empty_system_ne_generic_assistant": es_tokens != ga_tokens,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
