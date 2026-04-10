"""4-tier discrete reward for InterCode-ALFA NL2Bash tasks.

Adapted from xyhu's src/rl/reward_intercode.py for use with rLLM multi-turn env.

Reward tiers:
  1.0 — Execution output exact match (stdout + filesystem)
  0.5 — Correct base command + key flags/args overlap (Jaccard >= 0.5)
  0.2 — Correct base command only (wrong or missing args)
  0.0 — Wrong base command, unparseable, or empty output

Short-circuit: if base command doesn't match, return 0.0 immediately
without container execution. Containers are only invoked when the command
reaches the 0.5 tier to check for execution output match → 1.0.

Supports gold2 (alternate gold answer) — takes the best score.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.grpo.container_pool import ContainerPair, GIT_RESET_CMD, udocker_exec

log = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Structured reward output."""
    total: float = 0.0
    match_type: str = "none"  # none, base_cmd, flags, execution, command
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------
def parse_commands(text: str) -> List[str]:
    """Extract bash commands from model output.

    Looks for markdown code blocks, $ lines, or bare-command fallback.
    Strips <think>...</think> reasoning blocks and special tokens.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    commands = []

    # Markdown code blocks
    code_blocks = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line.lstrip("$ "))
    if commands:
        return commands

    # Lines starting with $
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("$ "):
            cmd = stripped[2:].strip()
            if cmd:
                commands.append(cmd)
    if commands:
        return commands

    # Bare-command fallback: first non-empty, non-comment line
    for line in text.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            commands.append(line)
    return commands


# ---------------------------------------------------------------------------
# Command structure parsing for tiered reward
# ---------------------------------------------------------------------------
def _split_on_pipes(cmd: str) -> List[str]:
    """Split command string on pipe '|', respecting quotes.

    Does NOT split on '||' (logical OR).
    """
    segments = []
    current: list = []
    i = 0
    in_single = False
    in_double = False
    while i < len(cmd):
        c = cmd[i]
        if c == "\\" and not in_single and i + 1 < len(cmd):
            current.append(c)
            current.append(cmd[i + 1])
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif c == "|" and not in_single and not in_double:
            if i + 1 < len(cmd) and cmd[i + 1] == "|":
                current.append("||")
                i += 2
                continue
            segments.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        segments.append("".join(current))
    return segments


def _normalize_flags(args: set) -> set:
    """Normalize flag sets: '-la' → {'-l', '-a'}."""
    normalized = set()
    for arg in args:
        if arg.startswith("--") or not arg.startswith("-") or len(arg) <= 1:
            normalized.add(arg)
        elif arg.startswith("-") and all(c.isalpha() for c in arg[1:]):
            for c in arg[1:]:
                normalized.add(f"-{c}")
        else:
            normalized.add(arg)
    return normalized


def parse_command_parts(cmd: str) -> List[Tuple[str, set]]:
    """Parse a shell command into [(base_command, {flags_and_args}), ...].

    Handles pipes by splitting on '|' first.
    Returns [] if empty or unparseable.
    """
    segments = _split_on_pipes(cmd.strip())
    result = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        base = tokens[0]
        if "/" in base:
            base = base.rsplit("/", 1)[-1]
        args = set(tokens[1:])
        result.append((base, args))
    return result


def compute_command_similarity(gold_cmd: str, generated_cmd: str) -> float:
    """Compare two commands structurally. Returns tier score (0.0, 0.2, 0.5).

    Does NOT check execution output — that's done separately for the 1.0 tier.
    For piped commands, requires ALL base commands to match for any score > 0.
    """
    gold_parts = parse_command_parts(gold_cmd)
    gen_parts = parse_command_parts(generated_cmd)

    if not gold_parts or not gen_parts:
        return 0.0

    if len(gold_parts) != len(gen_parts):
        return 0.0

    for (g_base, _), (p_base, _) in zip(gold_parts, gen_parts):
        if g_base != p_base:
            return 0.0

    # All base commands match → at least 0.2
    all_gold_flags: set = set()
    all_gen_flags: set = set()
    for (_, g_args), (_, p_args) in zip(gold_parts, gen_parts):
        all_gold_flags |= _normalize_flags(g_args)
        all_gen_flags |= _normalize_flags(p_args)

    # Both have no flags → 0.5 (e.g., "date" vs "date")
    if not all_gold_flags and not all_gen_flags:
        return 0.5

    # Generated has no flags but gold does → 0.5
    # (right command, missing flags, better than wrong flags)
    if not all_gen_flags and all_gold_flags:
        return 0.5

    union = all_gold_flags | all_gen_flags
    intersection = all_gold_flags & all_gen_flags
    jaccard = len(intersection) / len(union) if union else 1.0

    if jaccard >= 0.5:
        return 0.5
    return 0.2


# ---------------------------------------------------------------------------
# git status parsing
# ---------------------------------------------------------------------------
def parse_git_status(status_output: str) -> List[Tuple[str, str]]:
    """Parse `git status --short` into [(path, status_code), ...]."""
    changes = []
    for line in status_output.splitlines():
        line = line.rstrip()
        if not line or len(line) < 4:
            continue
        status_code = line[:2].strip()
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ")[-1]
        if path_part.startswith('"') and path_part.endswith('"'):
            path_part = path_part[1:-1]
        changes.append((path_part, status_code))
    return changes


# ---------------------------------------------------------------------------
# Execution comparison (agent vs gold via containers)
# ---------------------------------------------------------------------------
def _compare_execution(
    pair: ContainerPair,
    gold_cmd: str,
    generated_cmd: str,
) -> Dict:
    """Execute gold and generated commands in fresh containers, compare outputs.

    Only called when structural score >= 0.5 — checks for 1.0 upgrade.

    Returns dict with total, gold_output, agent_output, match_type.
    """
    shell = pair.shell
    env_vars = pair.env_vars
    result = {
        "total": 0.5,
        "gold_output": "",
        "agent_output": "",
        "match_type": "flags",
    }

    # Short-circuit: exact command string match → 1.0
    if gold_cmd.strip() == generated_cmd.strip():
        result["total"] = 1.0
        result["match_type"] = "command"
        return result

    # Reset eval container and run gold command
    udocker_exec(pair.eval_name, GIT_RESET_CMD, shell=shell, timeout_sec=30, env_vars=env_vars)
    gold_output, _ = udocker_exec(
        pair.eval_name, gold_cmd, shell=shell, timeout_sec=30, env_vars=env_vars,
    )
    result["gold_output"] = gold_output

    # Reset agent container and run generated command
    udocker_exec(pair.agent_name, GIT_RESET_CMD, shell=shell, timeout_sec=30, env_vars=env_vars)
    agent_output, _ = udocker_exec(
        pair.agent_name, generated_cmd, shell=shell, timeout_sec=30, env_vars=env_vars,
    )
    result["agent_output"] = agent_output

    # Check stdout match
    stdout_match = gold_output.strip() == agent_output.strip()

    # Check filesystem match
    agent_status, _ = udocker_exec(pair.agent_name, "git status --short", shell=shell, timeout_sec=10, env_vars=env_vars)
    eval_status, _ = udocker_exec(pair.eval_name, "git status --short", shell=shell, timeout_sec=10, env_vars=env_vars)
    fs_match = set(parse_git_status(agent_status)) == set(parse_git_status(eval_status))

    if stdout_match and fs_match:
        result["total"] = 1.0
        result["match_type"] = "execution"

    return result


# ---------------------------------------------------------------------------
# Main reward function (used by UdockerBashEnv)
# ---------------------------------------------------------------------------
def compute_tiered_reward(
    pair: ContainerPair,
    generated_cmd: str,
    gold: str,
    gold2: Optional[str] = None,
) -> RewardResult:
    """Compute 4-tier discrete reward for a generated bash command.

    Checks both gold and gold2 (alternate answer), returns the best score.

    For rLLM multi-turn: this is called after each agent step.
    The pair's containers are used for execution comparison when needed.

    Returns:
        RewardResult with total in {0.0, 0.2, 0.5, 1.0}
    """
    result = RewardResult()

    if not generated_cmd or not generated_cmd.strip():
        result.details["reason"] = "empty command"
        return result

    # Structural scoring against both golds
    struct1 = compute_command_similarity(gold, generated_cmd)
    struct2 = compute_command_similarity(gold2, generated_cmd) if gold2 and gold2 != gold else 0.0
    best_structural = max(struct1, struct2)

    if best_structural == 0.0:
        result.details["reason"] = "base command mismatch"
        return result

    if best_structural == 0.2:
        result.total = 0.2
        result.match_type = "base_cmd"
        result.details["struct_gold1"] = struct1
        result.details["struct_gold2"] = struct2
        return result

    # structural >= 0.5 → try execution comparison for 1.0 upgrade
    try:
        best_score = 0.0

        if struct1 >= 0.5:
            exec1 = _compare_execution(pair, gold, generated_cmd)
            best_score = max(best_score, exec1["total"])
            result.details["exec_gold1"] = exec1

        if best_score < 1.0 and gold2 and gold2 != gold and struct2 >= 0.5:
            exec2 = _compare_execution(pair, gold2, generated_cmd)
            best_score = max(best_score, exec2["total"])
            result.details["exec_gold2"] = exec2

        result.total = best_score
        result.match_type = "execution" if best_score >= 1.0 else "flags"

    except Exception as e:
        log.error("Execution comparison failed: %s", e)
        result.total = best_structural
        result.match_type = "flags"
        result.details["exec_error"] = str(e)

    return result
