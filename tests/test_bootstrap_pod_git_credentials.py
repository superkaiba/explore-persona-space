"""Tests for the ``scripts/bootstrap_pod.sh`` git credential handling (#1239).

Background: the pod bootstrap historically wrote a tokenized remote URL
(``https://x-access-token:${GITHUB_TOKEN}@github.com/...``) into the pod's
``.git/config`` and enabled ``git config --global credential.helper store``
(plaintext ``/root/.git-credentials``). #1239 ports the #1205 GCE fix to the
pod lane: a tokenless public remote plus an env-reading, ``.env``-fallback,
github.com-host-scoped credential helper (``$GIT_CRED_HELPER``), configured
repo-local (step 4, both branches) and global (step 7), with the ``store``
helper removed and its plaintext file scrubbed.

These tests are static + local-subprocess only (no live pod, no network):
they pin the no-token-at-rest invariant, the helper's shape (POSIX-sh, no
single quotes, host-scoped), the ``store`` removal + scrub, the existing-repo
retrofit, the double-quoted ssh_cmd context for the step-7 global config,
and — functionally — that the helper snippet, materialized via real bash and
executed the way git does (``sh -c '<body> "$@"' helper get``), emits the
token from the environment, from a ``.env`` fallback (plain and quoted
values), and degrades to an empty password (exit 0) when both are absent.

As of #1271 this file also pins the no-token-in-URL invariant on the
experimenter recipe surfaces (agent specs + the experimenter agent-memory).

As of #1401 the ``REPO_URL_TOKENLESS`` / ``GIT_CRED_HELPER`` definitions live
in the shared sourceable lib ``scripts/_git_cred_helper.sh`` (single
definition; ``bootstrap_pod.sh`` and ``sync_env_keys.sh`` both source it), so
the helper-assignment readers point at the lib while the bootstrap-text
assertions keep reading ``bootstrap_pod.sh``. The no-tokenized-URL scan
extends to the lib + ``sync_env_keys.sh``.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"
CRED_LIB = REPO_ROOT / "scripts" / "_git_cred_helper.sh"
SYNC_ENV_KEYS = REPO_ROOT / "scripts" / "sync_env_keys.sh"

_ASSIGN_PREFIX = 'GIT_CRED_HELPER="'
_HELPER_CONFIG_KEY = "credential.https://github.com.helper"


def _script_text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _helper_assignment_line() -> str:
    """Return the full ``GIT_CRED_HELPER="..."`` assignment line from the shared lib."""
    for line in CRED_LIB.read_text(encoding="utf-8").splitlines():
        if line.startswith(_ASSIGN_PREFIX):
            return line
    pytest.fail(f"no line starting with {_ASSIGN_PREFIX!r} in {CRED_LIB}")


def _helper_assignment_value_raw() -> str:
    """The raw (still-escaped) text between the assignment's outer double quotes."""
    # Sweep disposition (#2368): NOT a first-quote scan — this takes the whole
    # assignment LINE, asserts a terminal double quote, and slices prefix/suffix,
    # so embedded escaped quotes are INCLUDED in the value, never truncated; the
    # value is additionally cross-checked by real-bash materialization in test 5.
    line = _helper_assignment_line()
    assert line.endswith('"'), f"assignment must end with a closing double quote: {line!r}"
    return line[len(_ASSIGN_PREFIX) : -1]


def _single_quoted_ssh_cmd_blocks(text: str) -> list[str]:
    """Bodies of every single-quoted ``ssh_cmd '...'`` call in the script.

    Hardened per #2368 (same defect family as #2360 blockers B1/B5). A bash
    single-quoted STRING literal cannot contain a single quote, but the
    logical PAYLOAD handed to ``ssh_cmd`` can continue past the first ``'``
    via standard shell idioms: the quote-embed continuation
    (close-quote, backslash-escaped quote, reopen-quote), adjacent
    double-quote concatenation, or a same-line second quoted argument.
    A bare first-quote slice silently TRUNCATES all of these, and the
    negative assertion over the extracted blocks (test 4b) never inspects
    the uninspected tail — the test stays green while the tail regresses.

    Therefore, after closing each block at the first ``'`` following the
    opener, the extractor REQUIRES that closing quote to be followed by a
    NEWLINE or END-OF-TEXT — anything else is a loud ``pytest.fail`` naming
    the character offset and the following characters. This strictness
    deliberately also refuses a same-line second quoted argument
    (``ssh_cmd 'safe' 'tail'``): today ``ssh_cmd()`` forwards exactly
    ``"$1"`` (bootstrap_pod.sh, the 3-line helper above the argument
    parsing), so a second argument is dead text, but the guard must not
    DEPEND on that unpinned invariant — under a ``"$1"`` -> ``"$@"`` drift
    the second argument becomes live remote payload.

    Documented limitation: a future LEGITIMATE same-line shape (e.g. a real
    second argument) will fail here and requires a deliberate, reviewed
    edit of this terminator predicate — it is never absorbed silently.
    """
    blocks: list[str] = []
    idx = 0
    marker = "ssh_cmd '"
    while True:
        start = text.find(marker, idx)
        if start == -1:
            return blocks
        open_q = start + len(marker)
        close_q = text.index("'", open_q)
        after = text[close_q + 1 : close_q + 2]
        if after not in ("", "\n"):
            pytest.fail(
                f"ssh_cmd single-quoted block closing at character offset {close_q} is "
                f"followed by {text[close_q + 1 : close_q + 16]!r}, not newline/end-of-text "
                "— a same-line continuation would be silently truncated by a first-quote "
                "extractor; split the block or deliberately extend this terminator predicate"
            )
        block = text[open_q:close_q]
        # Defense in depth: trivially true under the first-quote slice above,
        # but pins the quote-free-interior invariant against a future rewrite.
        assert "'" not in block, "extracted ssh_cmd block interior must be quote-free"
        blocks.append(block)
        idx = close_q + 1


# ---------------------------------------------------------------------------
# 1. Durability pin: no tokenized remote URL anywhere in the script
# ---------------------------------------------------------------------------


def test_no_tokenized_remote_url_in_bootstrap() -> None:
    """No ``https://x-access-token:<tok>@github.com`` URL may reappear (#1239).

    ``x-access-token`` is permitted ONLY as the helper's
    ``echo username=x-access-token`` line — never as a URL userinfo prefix.
    Scan extends (#1401) to the shared cred-helper lib + sync_env_keys.sh.
    """
    import re

    for path in (BOOTSTRAP, CRED_LIB, SYNC_ENV_KEYS):
        text = path.read_text(encoding="utf-8")
        rel = path.name
        assert "https://x-access-token:" not in text, f"tokenized remote URL reintroduced: {rel}"
        assert not re.search(r"x-access-token:\S*@", text), (
            f"token-in-URL userinfo reintroduced: {rel}"
        )
        # Every remaining occurrence is the helper's username line.
        assert text.count("x-access-token") == text.count("username=x-access-token"), (
            f"x-access-token may appear only as `echo username=x-access-token` ({rel})"
        )


# ---------------------------------------------------------------------------
# 2. Helper shape: env-reading, .env fallback, host-scoped, quoting invariant
# ---------------------------------------------------------------------------


def test_credential_helper_host_scoped_env_reading() -> None:
    raw_value = _helper_assignment_value_raw()
    # Env-first read (escaped form in the double-quoted assignment).
    assert "\\${GITHUB_TOKEN:-}" in raw_value, "helper must read GITHUB_TOKEN from env first"
    # .env fallback sourced from the durable pod repo dir.
    assert "$REMOTE_DIR/.env" in raw_value, "helper must fall back to sourcing $REMOTE_DIR/.env"
    # Quoting invariant (§4.1): the stored value must contain NO single quote —
    # every use site wraps it in remote-level single quotes.
    assert "'" not in raw_value, "GIT_CRED_HELPER value must not contain a single quote"
    # No secret-shaped literal anywhere in the script or the shared lib.
    for scan in (_script_text(), CRED_LIB.read_text(encoding="utf-8")):
        assert "ghp_" not in scan and "github_pat_" not in scan, "secret-shaped literal"
    text = _script_text()
    # Host-scoped config key at all three sites: fresh-init, existing-repo
    # retrofit, and the step-7 global config.
    assert text.count(_HELPER_CONFIG_KEY) >= 3, (
        f"expected >=3 {_HELPER_CONFIG_KEY} config sites, found {text.count(_HELPER_CONFIG_KEY)}"
    )
    # Idempotent on the multi-valued helper key.
    assert text.count(f"--replace-all {_HELPER_CONFIG_KEY}") >= 3, (
        "helper config sets must use `git config --replace-all` (idempotent)"
    )


# ---------------------------------------------------------------------------
# 3. `credential.helper store` removed + plaintext file scrubbed
# ---------------------------------------------------------------------------


def test_credential_helper_store_absent() -> None:
    text = _script_text()
    assert "credential.helper store" not in text, (
        "the plaintext-at-rest `credential.helper store` line must stay removed"
    )
    assert "rm -f /root/.git-credentials" in text, (
        "step 7 must scrub the legacy /root/.git-credentials plaintext store"
    )
    assert "--unset-all credential.helper" in text, (
        "step 7 must unset the legacy unscoped credential.helper entries"
    )


# ---------------------------------------------------------------------------
# 4. Existing-repo branch retrofits the tokenless remote before the pull
# ---------------------------------------------------------------------------


def test_existing_repo_branch_retrofits_tokenless_remote() -> None:
    text = _script_text()
    start = text.index("if [ -d $REMOTE_DIR/.git ]")
    # Sweep disposition (#2368): first-OCCURRENCE keyword scan, same .index()
    # family as the ssh_cmd extractor, but every assertion on the slice below is
    # POSITIVE (must-contain / ordering via .index, which raises on absence), so
    # a truncated slice flips green->red (loud), never silently green.
    end = text.index("else", start)
    branch = text[start:end]
    assert "git remote set-url origin '$REPO_URL_TOKENLESS'" in branch, (
        "existing-repo branch must scrub a legacy tokenized remote via set-url"
    )
    assert _HELPER_CONFIG_KEY in branch, (
        "existing-repo branch must (re)install the credential helper"
    )
    # The retrofit must run BEFORE the pull so a legacy tokenized URL never
    # serves another fetch.
    assert branch.index("git remote set-url origin") < branch.index("git pull"), (
        "retrofit set-url must precede the pull"
    )


# ---------------------------------------------------------------------------
# 4b. Step-7 global helper lives in a DOUBLE-quoted ssh_cmd; the unexpanded
#     $GIT_CRED_HELPER never appears inside a single-quoted ssh_cmd block
# ---------------------------------------------------------------------------


def test_global_helper_config_in_double_quoted_ssh_cmd() -> None:
    text = _script_text()
    # The step-7 global set opens with `ssh_cmd "` so $GIT_CRED_HELPER expands
    # locally (a single-quoted block would store the literal variable name).
    assert (
        f"ssh_cmd \"git config --global --replace-all {_HELPER_CONFIG_KEY} '$GIT_CRED_HELPER'"
        in text
    ), "step-7 global helper config must live in a DOUBLE-quoted ssh_cmd call"
    for block in _single_quoted_ssh_cmd_blocks(text):
        assert "GIT_CRED_HELPER" not in block, (
            "unexpanded $GIT_CRED_HELPER inside a single-quoted ssh_cmd block "
            "(it would be stored literally, never expanded)"
        )


# ---------------------------------------------------------------------------
# 4c. Mutation tests (#2368): a payload continuing past the first closing
#     quote must FAIL extraction, never silently truncate
# ---------------------------------------------------------------------------

# Sentinel tail spliced past the first block's closing quote. It carries the
# exact literal the test-4b negative assertion scans for, so a truncating
# extractor that never surfaces it keeps that assertion green.
_MUTATION_SENTINEL = "echo $GIT_CRED_HELPER"

# Continuation shapes, each starting at the character immediately after the
# first single-quoted block's closing quote:
#   quote_embed      -> close-quote, backslash-escaped quote, reopen-quote
#                       ('a'\''b' — the #2360 idiom); next char: backslash.
#   double_quote     -> adjacent double-quoted string concatenation
#                       ('a'"b"); next char: double quote.
#   whitespace_arg   -> same-line SECOND quoted argument
#                       (ssh_cmd 'safe' 'tail'); next char: space.
_MUTATION_SPLICES = {
    "quote_embed": "\\''" + _MUTATION_SENTINEL + "'",
    "double_quote": '"' + _MUTATION_SENTINEL + '"',
    "whitespace_arg": " '" + _MUTATION_SENTINEL + "'",
}


@pytest.mark.parametrize("shape", sorted(_MUTATION_SPLICES))
def test_mutation_continuation_after_close_quote_fails_extraction(shape: str) -> None:
    """The hardened extractor FAILS on any same-line continuation splice.

    Red-direction evidence: under the OLD extractor (close at the first
    ``'``, no terminator check) every one of these splices yields a
    TRUNCATED block — the sentinel tail sits between the closing quote and
    the next ``ssh_cmd '`` marker, so it lands in NO extracted block, is
    never marker-matched, and the test-4b negative assertion
    (``"GIT_CRED_HELPER" not in block``) stays green: silent
    under-inspection. The hardened extractor instead ``pytest.fail``s the
    moment the closing quote is followed by anything but newline/end-of-text.
    """
    text = _script_text()
    marker = "ssh_cmd '"
    start = text.index(marker)
    # Sweep disposition (#2368): recorded-safe quote-char locator, NOT a payload
    # extractor. This first-quote .index() only LOCATES the splice point for the
    # mutation; the assert on the next line pins the newline precondition on the
    # real script's first block, and the mutated text is fed to the HARDENED
    # extractor (_single_quoted_ssh_cmd_blocks), which pytest.fails loud on any
    # same-line continuation — so a mislocated close quote here cannot silently
    # truncate inspected payload.
    close_q = text.index("'", start + len(marker))
    assert text[close_q + 1 : close_q + 2] == "\n", "real script's first block must be clean"
    mutated = text[: close_q + 1] + _MUTATION_SPLICES[shape] + text[close_q + 1 :]
    with pytest.raises(pytest.fail.Exception, match="followed by"):
        _single_quoted_ssh_cmd_blocks(mutated)


# ---------------------------------------------------------------------------
# 5. Functional: the helper snippet works under plain sh (the way git runs it)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("bash") is None,
    reason="sh/bash not available",
)
def test_helper_snippet_functional_under_sh(tmp_path: Path) -> None:
    """Materialize $GIT_CRED_HELPER via real bash, execute it the way git does.

    git runs a ``!``-prefixed helper as ``sh -c '<body> "$@"' <argv0> get``.
    Matrix: env token / .env fallback / quoted-.env fallback / neither
    (empty password, exit 0 — fail-loud at push, not a hang).
    """
    # Materialize the stored helper VALUE with real bash so the test never
    # re-implements the script's unescape rules (\" -> ", \$ -> $).
    assignment = _helper_assignment_line()
    materialize = subprocess.run(
        [
            "bash",
            "-c",
            f'REMOTE_DIR={shlex.quote(str(tmp_path))}; {assignment}; printf %s "$GIT_CRED_HELPER"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    helper_value = materialize.stdout
    assert helper_value.startswith("!"), f"helper must be a !-prefixed snippet: {helper_value!r}"
    assert "'" not in helper_value, "materialized helper value must contain no single quote"
    body = helper_value[1:]

    env_file = tmp_path / ".env"

    def run_helper(github_token: str | None) -> str:
        env = {"PATH": "/usr/bin:/bin"}
        if github_token is not None:
            env["GITHUB_TOKEN"] = github_token
        result = subprocess.run(
            ["sh", "-c", body + ' "$@"', "helper", "get"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"helper must exit 0 (fail-loud is git's job on empty password): {result.stderr}"
        )
        return result.stdout

    # Case 1: token in the invoking environment, no .env present.
    assert not env_file.exists()
    out = run_helper("envtok")
    assert "username=x-access-token" in out
    assert "password=envtok" in out

    # Case 2: env unset, plain KEY=value .env fallback.
    env_file.write_text("GITHUB_TOKEN=filetok\n", encoding="utf-8")
    out = run_helper(None)
    assert "password=filetok" in out

    # Case 3: env unset, QUOTED value (source-parity with the old
    # `set -a; . .env` semantics — a sed-style extractor would keep quotes).
    env_file.write_text('GITHUB_TOKEN="filetok"\n', encoding="utf-8")
    out = run_helper(None)
    assert "password=filetok" in out

    # Case 4: neither env nor .env — empty password, exit 0, no hang/crash.
    env_file.unlink()
    out = run_helper(None)
    assert "username=x-access-token" in out
    lines = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    assert lines.get("password", "MISSING") == "", (
        f"expected empty password when no token is available, got {lines!r}"
    )


# ---------------------------------------------------------------------------
# 6. Durability pin (#1271): no tokenized remote URL in the experimenter
#    salvage recipes (agent specs + the experimenter's always-loaded memory)
# ---------------------------------------------------------------------------


def _experimenter_recipe_surfaces() -> list[Path]:
    agents = sorted((REPO_ROOT / ".claude" / "agents").glob("*.md"))
    memory = sorted((REPO_ROOT / ".claude" / "agent-memory" / "experimenter").glob("*.md"))
    assert agents, "no agent specs found — repo layout changed?"
    assert memory, "no experimenter agent-memory files found — repo layout changed?"
    return agents + memory


def test_no_tokenized_remote_url_in_experimenter_recipes() -> None:
    """The #1239 no-token-in-URL invariant extends to the agent recipe
    surfaces (#1271): ``x-access-token`` may appear only as the credential
    helper's ``username=x-access-token`` line — never as URL userinfo
    (``x-access-token:<tok>@``) prescribing a tokenized remote URL.

    NOTE: the userinfo regex is SELF-MATCHING — in-scope docs must describe
    the banned form by paraphrase ("token-in-URL userinfo"), never by
    quoting the literal or the regex. Scope is the DOCUMENTED old form
    only: token-as-username variants (``https://ghp_...@`` / ``oauth2:...@``)
    are deliberately not covered (same class boundary as the #1239 pin).
    """
    import re

    for path in _experimenter_recipe_surfaces():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        assert "https://x-access-token:" not in text, f"tokenized remote URL in {rel}"
        assert not re.search(r"x-access-token:\S*@", text), f"token-in-URL userinfo in {rel}"
