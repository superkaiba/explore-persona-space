"""Pin: the fits phase-fingerprint embeds the REAL git commit sha (c24).

Regression for a latent unit-2 bug found by the unit-3 resume-matrix smoke:
``_phase_fingerprint`` read ``git_provenance().sha`` — an attribute that does
not exist (the field is ``commit_sha``) — so the ``except Exception`` degrade
branch (meant for git-less SLURM scratch trees) fired on EVERY run and pinned
``git_sha="unavailable-no-git-checkout"``, making the fingerprint's
code-identity leg permanently inert (a code change never forced a recompute).
This test fails pre-fix (both fingerprints equal under different shas) and
passes post-fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_fits as fits  # noqa: E402


def _args() -> object:
    return fits.build_argparser().parse_args(["--phase", "groups"])


def test_fingerprint_embeds_real_commit_sha(monkeypatch) -> None:
    from explore_persona_space.orchestrate import provenance

    def _fake(sha: str):
        return provenance.GitProvenance(commit_sha=sha, dirty=False, dirty_paths=[])

    monkeypatch.setattr(provenance, "git_provenance", lambda cwd=None: _fake("aaaa111"))
    fp_a = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    monkeypatch.setattr(provenance, "git_provenance", lambda cwd=None: _fake("bbbb222"))
    fp_b = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    assert fp_a != fp_b, (
        "phase fingerprint did not change with the git commit sha — the c24 "
        "code-identity leg is inert (the degrade branch is swallowing it)"
    )


def test_fingerprint_still_keys_inputs_and_flags() -> None:
    a = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    b = fits._phase_fingerprint(_args(), "groups", {"m": "y"})
    assert a != b
    args2 = fits.build_argparser().parse_args(["--phase", "groups", "--n-folds", "4"])
    c = fits._phase_fingerprint(args2, "groups", {"m": "x"})
    assert a != c
