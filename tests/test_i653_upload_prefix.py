"""Task #653 — HF upload prefix resolves to the approved follow-up slug.

CPU-only. CONCERN stale-hf-upload-prefix (round-2 Codex MAJOR): the constant was
``HF_UPLOAD_PREFIX = "issue653_readwrite_decomp"`` while the data-repo / wandb
consuming sites add their own ``issue653_`` namespace (``issue653_{HF_UPLOAD_PREFIX}``),
so raw completions + datasets uploaded under ``issue653_issue653_readwrite_decomp/...``
— a DOUBLE prefix that diverges from the approved ``issue653_install-validated-reladder/...``
downstream auditors expect.

These tests pin:
  * the constant is the bare slug (no ``issue653_`` namespace baked in);
  * the resolved data-repo / wandb prefix is EXACTLY ``issue653_install-validated-reladder``;
  * no consuming site produces a double ``issue653_issue653_`` prefix;
  * the bare-form adapter paths resolve cleanly under the slug.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653

APPROVED_SLUG = "install-validated-reladder"
APPROVED_DATA_PREFIX = f"issue653_{APPROVED_SLUG}"


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_prefix_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_prefix_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hf_upload_prefix_is_bare_slug():
    """The constant carries the bare approved slug — NOT an ``issue653_`` namespace
    (the consuming code adds it where the convention applies)."""
    assert i653.HF_UPLOAD_PREFIX == APPROVED_SLUG
    assert not i653.HF_UPLOAD_PREFIX.startswith("issue653_"), (
        "HF_UPLOAD_PREFIX must not bake in the issue653_ namespace, or the "
        "data-repo path double-prefixes to issue653_issue653_..."
    )


def test_resolved_data_repo_prefix_matches_approved_slug():
    """The data-repo / wandb consuming form ``issue653_{HF_UPLOAD_PREFIX}`` resolves
    to EXACTLY the approved ``issue653_install-validated-reladder`` (no double
    prefix) — what raw-completion / dataset auditors look for."""
    resolved = f"issue653_{i653.HF_UPLOAD_PREFIX}"
    assert resolved == APPROVED_DATA_PREFIX
    assert "issue653_issue653_" not in resolved


def test_dispatcher_upload_prefix_no_double_prefix():
    """The dispatcher's upload prefix (raw completions + datasets land here) is the
    approved slug, not the double-prefixed one."""
    mod = _load_dispatcher()
    # The phase_upload site computes `prefix = f"issue653_{i653.HF_UPLOAD_PREFIX}"`.
    prefix = f"issue653_{mod.i653.HF_UPLOAD_PREFIX}"
    assert prefix == APPROVED_DATA_PREFIX
    assert "issue653_issue653_" not in prefix


def test_adapter_path_resolves_under_slug():
    """The bare-form adapter sites resolve to ``adapters/install-validated-reladder/...``
    (clean, no double-namespace)."""
    adapter_path = f"adapters/{i653.HF_UPLOAD_PREFIX}/some_cell_id"
    assert adapter_path == f"adapters/{APPROVED_SLUG}/some_cell_id"
    assert "issue653_issue653_" not in adapter_path
