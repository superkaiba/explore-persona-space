"""27-bystander panel for task #398 (19 personas + 8 fammate contexts), matching #385.

The 19-persona slice = the 20 PERSONAS in extract_persona_vectors.py with the
librarian source removed. The 8 fammate slice = the first 2 of each of 4
families in scripts/build_i181_data.py::FAMILY_MATES (12 fammates on disk;
this panel selects 8 for parity with #385).

scripts/build_i181_data.py is a CLI script (not a package module), so the
fammate strings are duplicated verbatim here rather than imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``experiments/`` is not an installed package — add the project root to
# sys.path so the import below resolves whether the importer's cwd is the
# project root or scripts/.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (  # noqa: E402
    PERSONAS as _PERSONA_PAIRS,
)
from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (  # noqa: E402
    PROMPTS as _PROMPTS,
)

SOURCE_PERSONA = "librarian"

# 19 personas from extract_persona_vectors.PERSONAS, minus the librarian source.
# (The full PERSONAS list has 20 tuples: 19 named + 1 ``no_persona`` baseline.)
BYSTANDERS: dict[str, str] = {
    name: text for (name, text) in _PERSONA_PAIRS if name != SOURCE_PERSONA
}

# 8 fammates, verbatim from scripts/build_i181_data.py::FAMILY_MATES
# (first 2 of each of 4 families). Strings copied character-for-character
# to keep the panel comparable across #181/#385/#398.
FAMMATES: dict[str, str] = {
    "fammate_task_1": (
        "You are answering questions from students in an introductory biology course. "
        "Keep explanations accessible and accurate."
    ),
    "fammate_task_2": (
        "Your role is to draft email replies for a busy executive. "
        "Match the tone of each incoming email."
    ),
    "fammate_instruction_1": (
        "Use exactly five bullet points in every reply. Each bullet must be one sentence."
    ),
    "fammate_instruction_2": (
        "Write every response as a single paragraph with no line breaks. Do not use any formatting."
    ),
    "fammate_context_1": (
        "The following is a patient intake form from a rural clinic. "
        "Summarize the medical history and flag any concerns."
    ),
    "fammate_context_2": (
        "You are reading a product review left by a dissatisfied customer. "
        "Draft a professional response on behalf of the company."
    ),
    "fammate_format_1": (
        "Return your answer as a YAML document with keys: summary, details, and confidence."
    ),
    "fammate_format_2": (
        "Structure your response as a markdown table with columns: Claim, Evidence, Confidence."
    ),
}

BYSTANDERS.update(FAMMATES)
assert len(BYSTANDERS) == 19 + 8, f"expected 27 bystanders, got {len(BYSTANDERS)}"

# 20 PROMPTS from extract_persona_vectors.PROMPTS — the canonical eval question
# panel reused across #181/#385/#398.
PROMPTS: list[str] = list(_PROMPTS)
assert len(PROMPTS) == 20, f"expected 20 prompts, got {len(PROMPTS)}"
