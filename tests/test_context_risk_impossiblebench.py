from __future__ import annotations

import scripts.context_risk_impossiblebench as impossible


def test_find_code_matches_last_block_convention():
    completion = "first\n```\ndef f(): return 1\n```\nthen\n```\ndef f(): return 2\n```"
    assert impossible.find_code(completion).strip() == "def f(): return 2"


def test_verification_program_places_tests_after_answer():
    row = {
        "prompt": "def f():\n    pass",
        "test": "def check(candidate):\n    assert candidate() == 2",
        "entry_point": "f",
    }
    program = impossible.verification_program(row, "```python\ndef f(): return 2\n```")
    assert program.index("def f(): return 2") < program.index("def check(candidate)")
    assert program.rstrip().endswith("check(f)")
