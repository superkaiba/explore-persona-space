"""Issue #262 — EM-first persona-flattening test (librarian source).

Tests whether emergent misalignment (EM) flattens persona space, by comparing
post-coupling marker leakage on bystander personas across 4 conditions:

    C1 base-first        : raw Instruct -> couple librarian+[ZLT]
    C2 EM-first (primary): raw -> EM LoRA -> couple librarian+[ZLT]
    C2' cross-source     : EM-merged but couple on BASE-generated data
                          (artifact-detection arm for H3)
    C3 benign-first      : raw -> benign Tulu-3 SFT -> couple librarian+[ZLT]

Plan: .claude/plans/issue-262.md (cached, gitignored — read from the
parent worktree at /home/thomasjiralerspong/explore-persona-space/).

Sibling: scripts/plot_issue262.py builds the hero figure + diagnostic panels.

Note: implementation lands in subsequent commits on the issue-262 branch;
this stub exists so the PR can be opened.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Implementation pending — see .claude/plans/issue-262.md "
        "(experiment-implementer dispatched after PR creation)."
    )


if __name__ == "__main__":
    main()
