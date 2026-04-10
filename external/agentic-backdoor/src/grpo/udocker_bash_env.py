"""Udocker-based environment for InterCode-ALFA GRPO training via rLLM.

Uses pre-created container pairs (agent + eval) from ContainerPool.
Follows the InterCode interaction model for bash:
  1. Agent receives task instruction
  2. Agent outputs a bash command (raw text, no XML tags)
  3. Env executes the command and computes tiered reward
  4. If reward == 1.0, episode ends (perfect score)
  5. If reward < 1.0, agent sees the output and can retry next turn
  6. At max_steps, episode ends with last reward

Reward: 4-tier discrete {0.0, 0.2, 0.5, 1.0}
  - Structural command similarity (fast, no containers)
  - Execution comparison when structural >= 0.5 (containers)
  - Supports alternate gold (gold2) — takes best score

The model outputs bare bash commands (matching SFT training data).
No XML tags, no "submit" keyword.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv  # type: ignore

from src.grpo.container_pool import (
    ContainerPair,
    ContainerPool,
    GIT_RESET_CMD,
    udocker_exec,
)
from src.grpo.rewards.nl2bash_reward import (
    RewardResult,
    compute_tiered_reward,
    parse_commands,
)
from src.tbench_rllm.agent_env_dtos import ActionPayload, StepObservation

logger = logging.getLogger(__name__)


@dataclass
class InterCodeTask:
    """Configuration for an InterCode-ALFA task, loaded from parquet extra_info."""
    index: int
    local_index: int
    container_num: int
    query: str
    gold: str
    gold2: str
    difficulty: int


class UdockerBashEnv(BaseEnv):
    """InterCode-ALFA bash environment using udocker container pairs.

    Interaction model (matches InterCode eval_n_turn.py):
      - Agent outputs raw bash command (no XML, no "submit")
      - Env executes it in the agent container
      - Env computes tiered reward (structural + execution comparison)
      - If reward == 1.0 → episode ends (perfect, stop early)
      - If reward < 1.0 → agent sees output, can retry next turn
      - At max_steps → episode ends with last reward
    """

    def __init__(
        self,
        task: InterCodeTask,
        env_id: Optional[str] = None,
    ):
        self.task = task
        self.env_id = env_id or str(uuid.uuid4())

        # Container pair (checked out from pool on reset)
        self.pair: Optional[ContainerPair] = None
        self.done = False
        self.steps_taken = 0
        self.last_observation = ""
        self.last_reward: Optional[float] = None
        self.max_reward: float = 0.0  # Best reward seen across all steps
        self.best_step: int = 0       # Step at which max_reward was achieved

    def reset(self) -> Tuple[Dict, Dict]:
        """Reset environment: checkout container pair, return task instruction."""
        logger.info(
            "[%s] Reset for task idx=%d (container=%d, difficulty=%d)",
            self.env_id[:8], self.task.index, self.task.container_num,
            self.task.difficulty,
        )

        # Return any existing pair to pool
        if self.pair:
            self._cleanup()

        # Checkout a container pair from the pool
        pool = ContainerPool.get_instance()
        try:
            self.pair = pool.checkout(self.task.container_num, timeout=600)
        except (TimeoutError, Exception) as e:
            logger.warning("[%s] Container checkout failed: %s", self.env_id[:8], e)
            self.pair = None
            # Do NOT set done=True here — agent must generate at least one
            # response token or trajectory transform crashes (empty response).
            # step() will handle the missing pair and return done=True.
            self.last_reward = 0.0
            obs = StepObservation(msg=self.task.query, status="success")
            return obs.model_dump(), {}

        # Reset both containers to clean state (git reset, or full restore if corrupted)
        if not pool.reset_container(self.pair):
            logger.warning("[%s] Container reset failed, releasing pair", self.env_id[:8])
            pool.checkin(self.pair)
            self.pair = None
            self.last_reward = 0.0
            obs = StepObservation(msg=self.task.query, status="success")
            return obs.model_dump(), {}

        # Reset state
        self.done = False
        self.steps_taken = 0
        self.last_observation = ""
        self.last_reward = None
        self.max_reward = 0.0
        self.best_step = 0

        # Return task instruction as initial observation
        obs = StepObservation(
            msg=self.task.query,
            status="success",
        )
        return obs.model_dump(), {}

    def step(self, action: Dict) -> Tuple[Any, float, bool, Dict]:
        """Execute one step: extract command, run it, compute reward.

        Following InterCode eval_n_turn.py:
          1. Extract bash command from model's raw output
          2. Execute in agent container
          3. Compute tiered reward (structural + execution)
          4. If reward >= 1.0 → done=True (perfect, stop early)
          5. Otherwise → done=False, agent sees output and can retry
        """
        if self.done:
            logger.warning("[%s] Step on completed env", self.env_id[:8])
            return {}, 0.0, True, {"error": "Environment is done"}

        # If container checkout failed in reset(), end immediately with 0 reward.
        # This path ensures the trajectory has at least one response token.
        if self.pair is None:
            logger.warning("[%s] No container pair, ending with 0 reward", self.env_id[:8])
            self.done = True
            self.last_reward = 0.0
            obs = StepObservation(msg="Container unavailable", status="error")
            return obs.model_dump(), 0.0, True, {"error": "container_checkout_failed"}

        # Extract the raw command from the action payload
        action_payload = ActionPayload.model_validate(action)
        raw_response = action_payload.recent_model_resp or ""
        command = self._extract_command(raw_response)

        self.steps_taken += 1
        logger.debug("[%s] Step %d: %s", self.env_id[:8], self.steps_taken, command[:80])

        # Execute the command in the agent container
        output, returncode = udocker_exec(
            self.pair.agent_name, command,
            shell=self.pair.shell, timeout_sec=30,
            env_vars=self.pair.env_vars,
        )
        self.last_observation = output

        # Compute tiered reward using both containers
        reward_result = self._compute_reward(command)
        reward = reward_result.total
        self.last_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward
            self.best_step = self.steps_taken

        # Early stop if perfect reward (InterCode: "if reward == 1: break")
        if reward >= 0.99:
            self.done = True
            logger.info(
                "[%s] Perfect reward %.3f at step %d (match=%s), stopping early",
                self.env_id[:8], reward, self.steps_taken, reward_result.match_type,
            )

        obs = StepObservation(
            msg=output if output.strip() else "(no output)",
            status="success" if returncode == 0 else "error",
        )

        info = {
            "reward_details": {
                "total": reward,
                "match_type": reward_result.match_type,
                "details": reward_result.details,
            }
        }

        return obs.model_dump(), reward, self.done, info

    def compute_final_reward(self) -> float:
        """Called by rLLM at trajectory end (max_steps/timeout/truncation).

        Returns the best reward seen across all steps, decayed by which step
        achieved it. Solving on step 1 gives full reward; later steps get a
        discount of 0.1 per step. This incentivizes getting it right early
        while still rewarding successful retries.

        For 1-turn (max_steps=1): no decay, equivalent to raw reward.
        For 5-turn: step1=1.0x, step2=0.9x, step3=0.8x, step4=0.7x, step5=0.6x
        """
        STEP_DECAY = float(os.environ.get("GRPO_STEP_DECAY", "0.1"))

        if self.last_reward is None:
            # No steps taken — compute reward of current state
            reward_result = self._compute_reward("")
            self.last_reward = reward_result.total
            if self.last_reward > self.max_reward:
                self.max_reward = self.last_reward
                self.best_step = 0

        decay = 1.0 - STEP_DECAY * max(0, self.best_step - 1)
        final = self.max_reward * max(decay, 0.0)

        logger.info(
            "[%s] Final reward: %.3f (max=%.3f at step %d, decay=%.1f, steps=%d)",
            self.env_id[:8], final, self.max_reward, self.best_step, decay, self.steps_taken,
        )
        return final

    def close(self):
        """Return container pair to pool."""
        logger.debug("[%s] Closing", self.env_id[:8])
        self._cleanup()

    @staticmethod
    def from_dict(info: Dict) -> "UdockerBashEnv":
        """Create environment from parquet extra_info dict.

        Expects InterCode-ALFA fields:
          index, local_index, container_num, query, gold, gold2, difficulty
        """
        task = InterCodeTask(
            index=info.get("index", 0),
            local_index=info.get("local_index", 0),
            container_num=info["container_num"],
            query=info["query"],
            gold=info["gold"],
            gold2=info.get("gold2", info["gold"]),
            difficulty=info.get("difficulty", -1),
        )

        return UdockerBashEnv(
            task=task,
            env_id=info.get("env_id"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # --- Private methods ---------------------------------------------------

    def _extract_command(self, response: str) -> str:
        """Extract a bash command from the model's raw response.

        Uses parse_commands from the reward module (handles thinking tags,
        code blocks, special tokens, bare commands).
        """
        commands = parse_commands(response)
        return commands[0] if commands else response.strip()

    def _compute_reward(self, generated_cmd: str) -> RewardResult:
        """Compute the 4-tier reward using container pair."""
        if not self.pair:
            logger.error("[%s] No container pair for reward computation", self.env_id[:8])
            return RewardResult()

        try:
            result = compute_tiered_reward(
                pair=self.pair,
                generated_cmd=generated_cmd,
                gold=self.task.gold,
                gold2=self.task.gold2,
            )
            logger.info(
                "[%s] Reward: %.1f (match=%s) cmd='%s' gold='%s'",
                self.env_id[:8], result.total, result.match_type,
                generated_cmd[:60], self.task.gold[:60],
            )
            return result
        except Exception as e:
            logger.error("[%s] Reward calc failed: %s", self.env_id[:8], e)
            return RewardResult()

    def _cleanup(self) -> None:
        """Return the container pair to the pool."""
        if self.pair:
            try:
                ContainerPool.get_instance().checkin(self.pair)
            except Exception as e:
                logger.debug("[%s] Pool checkin error: %s", self.env_id[:8], e)
            self.pair = None
