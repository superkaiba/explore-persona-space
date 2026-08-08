"""Pin: #1739 many-small-file HF staging scripts force the PLAIN download path.

The 458-small-npz restore storm (2026-07-30) WEDGES xet_get indefinitely
(att-20260730-055211-syc, py-spy-confirmed) and errors hf_transfer
(att-20260730-063858-syc); the plain path handles small files fine. Both #1739
staging helpers must therefore set ``HF_HUB_DISABLE_XET=1`` +
``HF_HUB_ENABLE_HF_TRANSFER=0`` in-script BEFORE any huggingface_hub-reaching
import (``huggingface_hub`` freezes env at import) — the gotchas "HF Hub
download-accelerator FAILURE MATRIX" rule for small-file-storm scripts.
Source-order pin (mechanizable check from code-review v11 on task #1739).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE_SCRIPTS = [
    "scripts/issue1739_core_tail_stage.py",
    "scripts/issue1739_restore_partial.py",
]

# First import that reaches huggingface_hub: direct, or via orchestrate.hub
# (which imports huggingface_hub transitively at module load).
_HUB_IMPORT = re.compile(
    r"^\s*(?:import\s+huggingface_hub"
    r"|from\s+huggingface_hub\s+import"
    r"|from\s+explore_persona_space\.orchestrate(?:\.hub)?\s+import)",
    re.M,
)


def test_stage_scripts_disable_accelerators_before_hub_import() -> None:
    """Both env disables exist and precede the first hub-reaching import."""
    for rel in STAGE_SCRIPTS:
        src = (REPO_ROOT / rel).read_text()
        xet = re.search(r'^os\.environ\["HF_HUB_DISABLE_XET"\]\s*=\s*"1"', src, re.M)
        hft = re.search(r'^os\.environ\["HF_HUB_ENABLE_HF_TRANSFER"\]\s*=\s*"0"', src, re.M)
        assert xet and hft, f"{rel}: missing in-script HF accelerator disables (#1739 xet wedge)"
        hub_import = _HUB_IMPORT.search(src)
        assert hub_import, f"{rel}: no huggingface_hub-reaching import found — update this pin"
        assert max(xet.start(), hft.start()) < hub_import.start(), (
            f"{rel}: HF accelerator disables must precede the first "
            "huggingface_hub-reaching import (env frozen at import)"
        )
