"""Tests for :mod:`explore_persona_space.orchestrate.spawn_agent`.

Plan §3 Phase 4.5 of issue #320 — the GH_TOKEN env scrub. The contract:

* ``GH_TOKEN`` and ``GITHUB_TOKEN`` MUST be removed from any subagent's
  env (closes the agent-context-window leakage path).
* Every other secret (WANDB_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY,
  OPENAI_API_KEY, RUNPOD_API_KEY, SUPABASE_*, etc.) MUST survive —
  subagents like ``analyzer`` and ``experimenter`` need them.
* The function MUST be pure (returns a new dict; ``parent_env``
  unchanged).
"""

from __future__ import annotations

import os

import pytest

from explore_persona_space.orchestrate.spawn_agent import (
    GITHUB_AUTH_VARS,
    scrub_subagent_env,
)


class TestGhTokenIsRemoved:
    """The load-bearing rule: GH_TOKEN never reaches a subagent."""

    def test_gh_token_filtered(self) -> None:
        env = {"GH_TOKEN": "ghp_REDACTED", "PATH": "/usr/bin"}
        result = scrub_subagent_env(env)
        assert "GH_TOKEN" not in result

    def test_github_token_filtered(self) -> None:
        # gh CLI / octokit / PyGithub fall back to GITHUB_TOKEN when
        # GH_TOKEN is missing; leaving it in defeats the §3 contract.
        env = {"GITHUB_TOKEN": "ghp_REDACTED", "PATH": "/usr/bin"}
        result = scrub_subagent_env(env)
        assert "GITHUB_TOKEN" not in result

    def test_both_github_vars_filtered_together(self) -> None:
        env = {
            "GH_TOKEN": "ghp_x",
            "GITHUB_TOKEN": "ghp_y",
            "WANDB_API_KEY": "w",
        }
        result = scrub_subagent_env(env)
        assert "GH_TOKEN" not in result
        assert "GITHUB_TOKEN" not in result

    def test_github_auth_vars_constant_pinned(self) -> None:
        # Future contributors must update the constant + docstring +
        # CLAUDE.md note when adding/removing vars; this test pins the
        # current set so an accidental edit is caught.
        assert frozenset({"GH_TOKEN", "GITHUB_TOKEN"}) == GITHUB_AUTH_VARS


class TestOtherSecretsSurvive:
    """Non-GitHub secrets MUST pass through — subagents need them."""

    @pytest.mark.parametrize(
        "key",
        [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "RUNPOD_API_KEY",
            "PROJECT_PAT",
            "SUPABASE_ACCESS_TOKEN",
            "CODECOV_TOKEN",
        ],
    )
    def test_secret_survives(self, key: str) -> None:
        env = {key: "secret-value", "GH_TOKEN": "should-be-stripped"}
        result = scrub_subagent_env(env)
        assert result.get(key) == "secret-value"
        assert "GH_TOKEN" not in result

    def test_non_secret_env_passes_through(self) -> None:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/root",
            "HF_HOME": "/workspace/.cache/huggingface",
        }
        result = scrub_subagent_env(env)
        assert result == env

    def test_full_realistic_env(self) -> None:
        # Approximation of /issue's real spawn env: lots of unrelated
        # vars plus the GitHub one. The scrub must do nothing except
        # remove the one var.
        env = {
            "GH_TOKEN": "ghp_REDACTED",
            "WANDB_API_KEY": "w",
            "HF_TOKEN": "h",
            "ANTHROPIC_API_KEY": "a",
            "OPENAI_API_KEY": "o",
            "RUNPOD_API_KEY": "r",
            "PATH": "/usr/bin",
            "HOME": "/root",
            "USER": "thomasjiralerspong",
            "HF_HOME": "/workspace/.cache/huggingface",
        }
        result = scrub_subagent_env(env)
        assert "GH_TOKEN" not in result
        for k in env:
            if k != "GH_TOKEN":
                assert result.get(k) == env[k]


class TestPurity:
    """No side effects on the input mapping."""

    def test_returns_new_dict(self) -> None:
        env: dict[str, str] = {"GH_TOKEN": "x", "WANDB_API_KEY": "w"}
        result = scrub_subagent_env(env)
        # Different object identity.
        assert result is not env
        # Original dict is untouched.
        assert env == {"GH_TOKEN": "x", "WANDB_API_KEY": "w"}

    def test_accepts_os_environ(self) -> None:
        # Direct ``os.environ`` is a Mapping[str, str] but not a dict;
        # the scrub must accept it without complaint.
        result = scrub_subagent_env(os.environ)
        assert isinstance(result, dict)
        # Sanity: PATH should be present in any test environment.
        assert "PATH" in result
        # And no GH_TOKEN regardless of whether the test runner had it.
        assert "GH_TOKEN" not in result
        assert "GITHUB_TOKEN" not in result

    def test_empty_env(self) -> None:
        assert scrub_subagent_env({}) == {}

    def test_only_filtered_vars(self) -> None:
        env = {"GH_TOKEN": "x", "GITHUB_TOKEN": "y"}
        assert scrub_subagent_env(env) == {}
