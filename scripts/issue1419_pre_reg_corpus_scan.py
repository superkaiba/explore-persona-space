#!/usr/bin/env python
"""Corpus scan for the #1419 bare-'registered <noun>' `pre_reg` branch.

Re-runs the plan §5 precision measurement against the LIVE pattern in
`scripts/audit_clean_results_body_discipline.py` (no hardcoded copy to
drift): loads the audit module, splits `PATTERNS["pre_reg"]` into the
old alternation (through the `gate-pre-?registered` branch) and the new
bare-registered-noun branch (the alternation tail), rebuilds the exact
per-body scan source (`strip_code(strip_data_example_blocks(
strip_context_blockquotes(strip_frontmatter(body))))` →
`_restrict_pre_reg_to_prose_sections`), and counts new-branch matches
whose span does not overlap an old-alternation match over
`tasks/{completed,awaiting_promotion,followups_running}/*/body.md`
(paper stubs skipped). Prints
`bodies=N marginal_hits=N hit_bodies=N newline_bridges=N` plus per-body
samples. Re-run this after ANY change to the `pre_reg` pattern (plan
§7 kill criterion: non-jargon FPs ≤ 5, newline bridges = 0).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

_STATUSES = ("completed", "awaiting_promotion", "followups_running")
_OLD_TAIL_MARKER = "gate-pre-?registered"


def _load_audit_module():
    """Import the audit module from this script's own directory (the live tree)."""
    path = Path(__file__).resolve().parent / "audit_clean_results_body_discipline.py"
    spec = importlib.util.spec_from_file_location("audit_cr_body_discipline_1419", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _split_pattern(pattern: str) -> tuple[str, str]:
    """Split the live `pre_reg` alternation into (old branches, new branch)."""
    idx = pattern.index(_OLD_TAIL_MARKER)
    cut = idx + len(_OLD_TAIL_MARKER)
    if cut >= len(pattern) or pattern[cut] != "|":
        raise SystemExit(
            "pre_reg pattern shape changed: no alternation tail after "
            f"{_OLD_TAIL_MARKER!r} — update this script's split logic."
        )
    return pattern[:cut], pattern[cut + 1 :]


def _is_paper_stub(body: str) -> bool:
    """True when the frontmatter carries `paper: true` (paper stubs are skipped)."""
    if not body.startswith("---"):
        return False
    end = body.find("\n---", 3)
    fm = body[: end if end != -1 else len(body)]
    return bool(re.search(r"(?im)^paper\s*:\s*(?:true|'true'|\"true\")\s*$", fm))


def _default_repo_root() -> Path:
    """The MAIN checkout root (git-common-dir parent — worktree-safe)."""
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent,
    )
    return Path(out.stdout.strip()).parent


def main() -> None:
    """Scan the promoted/parked task-body corpus and print the marginal-hit counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root whose tasks/ tree to scan (default: the MAIN checkout root).",
    )
    parser.add_argument("--samples-per-body", type=int, default=3)
    args = parser.parse_args()
    root = (args.repo_root or _default_repo_root()).resolve()

    audit = _load_audit_module()
    old_pat, new_branch = _split_pattern(audit.PATTERNS["pre_reg"][0])

    n_bodies = n_skipped = n_marginal = n_bridges = 0
    hit_bodies: list[tuple[str, list[str]]] = []
    for status in _STATUSES:
        for body_path in sorted((root / "tasks" / status).glob("*/body.md")):
            body = body_path.read_text(encoding="utf-8")
            if _is_paper_stub(body):
                n_skipped += 1
                continue
            n_bodies += 1
            cleaned = audit.strip_code(
                audit.strip_data_example_blocks(
                    audit.strip_context_blockquotes(audit.strip_frontmatter(body))
                )
            )
            scan = audit._restrict_pre_reg_to_prose_sections(body, cleaned)
            old_spans = [m.span() for m in re.finditer(old_pat, scan, re.IGNORECASE)]
            samples: list[str] = []
            for m in re.finditer(new_branch, scan, re.IGNORECASE):
                if any(m.start() < e and s < m.end() for s, e in old_spans):
                    continue
                n_marginal += 1
                if "\n" in m.group(0):
                    n_bridges += 1
                samples.append(m.group(0))
            if samples:
                rel = str(body_path.relative_to(root))
                hit_bodies.append((rel, samples[: args.samples_per_body]))

    print(f"scanned repo root: {root} (paper stubs skipped: {n_skipped})")
    print(
        f"bodies={n_bodies} marginal_hits={n_marginal} "
        f"hit_bodies={len(hit_bodies)} newline_bridges={n_bridges}"
    )
    for rel, samples in hit_bodies:
        print(f"  {rel} -> " + " | ".join(repr(s) for s in samples))


if __name__ == "__main__":
    main()
