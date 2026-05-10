---
name: Claude Code model & effort persistence
description: Correct zshrc + settings.json for Opus 4.7 1M context with max effort — settings.json env block alone is NOT sufficient; shell rc export is required
type: reference
---

**Complete working config for Opus 4.7 + 1M + max effort:**

1. `~/.claude/settings.json`:
   ```json
   {
     "model": "opus[1m]",
     "env": {
       "CLAUDE_CODE_EFFORT_LEVEL": "max"
     }
   }
   ```

2. `~/dotfiles/config/zsh/zshrc.sh`:
   ```bash
   export CLAUDE_CODE_EFFORT_LEVEL=max
   # Claude is launched via Happy wrapper; pass --effort flag through the alias
   # because env-var approach alone proved unreliable in practice on this machine.
   alias claude="happy claude --effort max"
   ```

**Empirical finding (2026-04-17):** env-var approaches (both settings.json env block AND shell rc export) did NOT reliably set max effort on this machine — likely because Claude is launched via the Happy wrapper (`happy claude`), which may not forward env in the expected order, or because Claude reads effort via a code path that ignores the env var. The only thing that worked was the `--effort max` CLI flag in the launch alias.

**Key rules (empirically verified 2026-04-17):**

- `model: "opus[1m]"` in settings.json — documented alias; pins Opus with 1M context window.
- Effort levels on Opus 4.7: `low`, `medium`, `high`, `xhigh`, `max`.
- `"effortLevel": "max"` as a top-level key in settings.json does NOT work — silently falls back to default (`xhigh` on Opus 4.7). Multiple open issues: #43322, #48051, #34837, #37303.
- Per docs, `CLAUDE_CODE_EFFORT_LEVEL` env var IS the documented way to persist `max`.
- **BUT** the `env` block in `settings.json` is NOT sufficient on its own: Claude Code reads its effort level from shell env **before** fully applying its own env block. The env block does propagate to subprocesses (Bash tool sees `CLAUDE_CODE_EFFORT_LEVEL=max`), but Claude Code itself has already locked in effort by then.
- Shell-exported env vars take precedence over settings.json env block (when both are set, shell wins).
- Therefore the env var MUST be exported in the parent shell's rc file for Claude Code to actually launch with max effort.

**Shell stack on this machine:** bash auto-switches to zsh on login (via `exec /usr/bin/zsh -l` in `~/.bashrc`). Claude is launched as `happy claude` (alias in zshrc) → `node` → `claude`. So zshrc is the right place to export.

**Why both layers:** settings.json env block is the right documented approach and still exports to subprocesses (tool calls, hooks). The zshrc export is the *additional* layer needed for Claude Code's own effort selection at startup. Belt-and-suspenders; don't remove either.

**How to apply:** When the user asks to set default model/effort, (a) ensure `model` + `env.CLAUDE_CODE_EFFORT_LEVEL` are in `~/.claude/settings.json`, AND (b) ensure `export CLAUDE_CODE_EFFORT_LEVEL=max` is in the parent shell rc (zshrc on this machine). Never set `effortLevel: "max"` as a top-level key in settings.json.
