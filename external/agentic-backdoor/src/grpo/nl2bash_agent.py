"""NL2Bash agent — thin subclass of TerminalBenchAgent with a shorter system prompt.

Only overrides __init__ to load nl2bash_system_prompt.md instead of
the full terminal-bench system_prompt.md. Everything else (conversation
management, trajectory tracking, action payloads) is inherited.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.tbench_rllm.terminal_agent import TerminalBenchAgent

logger = logging.getLogger(__name__)


class NL2BashAgent(TerminalBenchAgent):
    """Agent with NL2Bash-specific system prompt."""

    def __init__(self, agent_id: Optional[str] = None):
        # Skip TerminalBenchAgent.__init__ to avoid loading the wrong system prompt,
        # but call BaseAgent.__init__ via super().__init__ chain.
        super().__init__(agent_id=agent_id)
        # Override the system prompt loaded by TerminalBenchAgent.__init__
        self.system_prompt = self._load_nl2bash_prompt()
        # Re-reset to apply the new system prompt to convo_history
        self.reset()

    @staticmethod
    def _load_nl2bash_prompt() -> str:
        prompt_path = Path(__file__).parent / "nl2bash_system_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"NL2Bash system prompt not found: {prompt_path}")
