from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/issue2588_mapping_rank_vs_capability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "issue2588_mapping_rank_vs_capability", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_exact_spearman_constant_input_is_json_safe() -> None:
    result = MOD.exact_spearman_permutation([1, 2, 3], [5, 5, 5])

    assert result == {
        "n": 3,
        "rho": None,
        "two_sided_exact_permutation_p": None,
        "n_permutations": 0,
        "method": "undefined_constant_input",
    }
