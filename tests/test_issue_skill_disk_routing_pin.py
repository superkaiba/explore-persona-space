"""Pin the disk-routing element (5) of the compute-character pre-launch statement.

The Step 9a-ter "Compute-character pre-launch statement" in
`.claude/skills/issue/SKILL.md` is the compute-review surface for the
workflow's PLANNERLESS paths (9a-ter zero-GPU floor, the Step 9b same-issue
follow-up loop, and the CLAUDE.md "User-chat inline free analysis"
carve-out). Task #1393 added a 5th element — multi-GB download DISK ROUTING
— after an inline free-analysis round on #823 pulled 14 GB of HF tensors
onto the shared boot disk `/` (ENOSPC, orchestrator Bash output lost; the
recovery then hit a root-owned top-level `/mnt/eps-data` mkdir failure).

These tests pin, so they cannot silently drift:

1. SKILL.md element (5): the multi-GB threshold, the `df -P` filesystem
   probe, the headroom multiplier, the `/` + `/tmp` bans, the
   fresh-top-level `/mnt/eps-data/<dir>` ban, the per-issue
   `/mnt/eps-data/$USER/issue<N>_<slug>/` fallback, and the extended
   escape one-liner suffix ("no multi-GB staging" — a round may take the
   escape only when it has neither a fit/battery stage NOR multi-GB
   staging).
2. The CLAUDE.md "User-chat inline free analysis" carve-out mirror of the
   same disk-routing substance.
3. The Step 9b cross-reference counting FIVE elements (no stale "same four
   elements" remains).

Assertions run on whitespace-NORMALIZED file text so prose re-wrapping
never breaks a pin; each token is still a verbatim substring of the rule.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = ROOT / "CLAUDE.md"


def _normalized(path: Path) -> str:
    """File text with all whitespace runs collapsed to single spaces.

    The pinned tokens are multi-word prose fragments; SKILL.md wraps prose
    at ~78 columns, so a raw-substring pin would break on any innocent
    re-wrap. Collapsing whitespace makes the pins wrap-insensitive while
    keeping them verbatim in substance.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def test_9a_ter_disk_routing_element_pinned() -> None:
    text = _normalized(ISSUE_SKILL)
    # Element (5) exists in the canonical statement.
    assert "(5) for any stage that downloads" in text
    # The multi-GB threshold that arms the duty.
    assert "≥ ~5 GB" in text
    # The filesystem-resolution probe (state-independent of the #681 bind).
    assert "`df -P <path>`" in text
    # The headroom multiplier (secondary check). Pinned without the
    # multiplication sign (ruff RUF001 bans that ambiguous unicode char in
    # Python strings; the .md text spells the full "~1.5x" with it).
    assert "free headroom ≥ ~1.5" in text
    # The boot-disk + /tmp bans (routing clause — primary).
    assert "NEVER lands on `/` (the shared boot disk) or `/tmp/`" in text
    # The root-owned fresh-top-level ban (the incident's second failure).
    assert "NEVER a fresh top-level `/mnt/eps-data/<dir>`" in text
    # The established user-writable per-issue fallback on the data disk.
    assert "/mnt/eps-data/$USER/" in text
    # Cache threading so the hub cache follows the routed staging path.
    assert "`HF_HOME` / `local_dir`" in text
    # E1b: the escape one-liner covers download-only rounds too — a round
    # with a multi-GB download can no longer take the fit/battery escape.
    assert "compute-character: no fit/battery stages, no multi-GB staging" in text


def test_claude_md_carveout_disk_routing_pinned() -> None:
    text = _normalized(CLAUDE_MD)
    # Mirror-drift guards for the user-chat inline carve-out clause.
    assert "df -P" in text
    assert "/mnt/eps-data/$USER/issue" in text
    assert "≥ ~5 GB" in text
    assert "NEVER `/`, `/tmp/`" in text


def test_step9b_mirror_says_five_elements() -> None:
    text = _normalized(ISSUE_SKILL)
    assert "same five elements" in text
    assert "same four elements" not in text
