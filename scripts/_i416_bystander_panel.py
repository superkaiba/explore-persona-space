"""27-bystander panel for task #416 (19 personas + 8 fammate contexts).

Identical to ``scripts/_i398_bystander_panel.py`` in every respect EXCEPT
``SOURCE_PERSONA = "software_engineer"``. The ``BYSTANDERS`` dict-comprehension
``{name: text for (name, text) in _PERSONA_PAIRS if name != SOURCE_PERSONA}``
automatically excludes software_engineer (the new #416 source) and re-includes
librarian (the #398 source, now a bystander here) — keeping the 28-persona
panel (1 source + 27 bystanders) identical between #398 and #416. Only the
source/bystander label swap.

scripts/build_i181_data.py is a CLI script (not a package module), so the
fammate strings are duplicated verbatim here rather than imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ``experiments/`` is not an installed package — add the project root to
# sys.path so the import below resolves whether the importer's cwd is the
# project root or scripts/.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ``extract_persona_vectors.py`` has a top-level ``os.environ["CUDA_VISIBLE_DEVICES"] = "5"``
# side effect (originally to pin its own GPU at import time). Capture the
# pre-import value, do the import for PERSONAS/PROMPTS, then restore — otherwise
# every script that uses this panel module silently inherits CUDA_VISIBLE_DEVICES=5
# and crashes at adapter-load time on the 1xH100 epm-issue-416 pod.
_PRE_IMPORT_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (  # noqa: E402
    PERSONAS as _PERSONA_PAIRS,
)
from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (  # noqa: E402
    PROMPTS as _PROMPTS,
)

if _PRE_IMPORT_CVD is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_IMPORT_CVD

SOURCE_PERSONA = "software_engineer"

# 19 personas from extract_persona_vectors.PERSONAS, minus the software_engineer source.
# (The full PERSONAS list has 20 tuples: 19 named + 1 ``no_persona`` baseline.)
# Librarian (the #398 source) is now back in this set as a bystander.
BYSTANDERS: dict[str, str] = {
    name: text for (name, text) in _PERSONA_PAIRS if name != SOURCE_PERSONA
}

# 8 fammates, verbatim from scripts/build_i181_data.py::FAMILY_MATES
# (first 2 of each of 4 families). Strings copied character-for-character
# to keep the panel comparable across #181/#385/#398/#416.
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
# panel reused across #181/#385/#398/#416.
PROMPTS: list[str] = list(_PROMPTS)
assert len(PROMPTS) == 20, f"expected 20 prompts, got {len(PROMPTS)}"

# Re-export the full (name, text) list from extract_persona_vectors.PERSONAS so
# downstream consumers (e.g. scripts/smoke_i398_logp_check.py) can import it
# from this module and inherit the CUDA_VISIBLE_DEVICES scrub above instead of
# triggering the env leak by importing extract_persona_vectors directly.
PERSONAS: list[tuple[str, str]] = list(_PERSONA_PAIRS)
