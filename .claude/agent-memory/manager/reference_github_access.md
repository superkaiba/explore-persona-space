---
name: Pod Access — SSH MCP + Config + Git
description: SSH MCP server (mcp-ssh-manager) configured for pod1-pod4; prefer MCP tools over Bash SSH; also SSH config aliases and git credentials
type: reference
---

## SSH MCP Server (PRIMARY — use this)

`mcp-ssh-manager` is configured in `.claude/mcp.json` with all 4 pods as `pod1`–`pod4`. **Always prefer MCP tools over `Bash("ssh podN ...")`.**

### Key tools (load via ToolSearch → `select:ssh_execute,ssh_list_servers,...`)

| Tool | What it does |
|------|-------------|
| `ssh_execute` | Run command on one pod (params: `server`, `command`) |
| `ssh_list_servers` | Show all pods + connection status |
| `ssh_upload` / `ssh_download` | File transfer (replaces `scp`) |
| `ssh_health_check` | Full diagnostics: CPU, RAM, disk, network |
| `ssh_group_execute` | Run command on ALL pods at once |
| `ssh_process_manager` | List/kill processes by CPU/mem |
| `ssh_monitor` / `ssh_tail` | Continuous monitoring / log tailing |

### When to fall back to Bash SSH
- Interactive/streaming output (e.g., `tail -f` via Bash `ssh pod1 "tail -f ..."`)
- Complex piped one-liners
- TTY-required commands

### Installation location
- Server: `/home/thomasjiralerspong/.local/node_modules/mcp-ssh-manager/src/index.js`
- Tool config: `~/.ssh-manager/tools-config.json` (core + monitoring enabled, DB/backup/sessions disabled)
- Pod configs: env vars in `.claude/mcp.json`

## SSH Config Aliases (FALLBACK)

`~/.ssh/config` defines aliases `pod1`–`pod4` with all connection params. Use for Bash fallback:
- `ssh pod1 "nvidia-smi"`
- `scp pod1:/workspace/file ./local`

| Alias | Pod Name | GPUs |
|-------|----------|------|
| pod1 | thomas-rebuttals | 4x H200 SXM |
| pod2 | thomas-rebuttals-2 | 8x H100 SXM 80GB |
| pod3 | thomas-rebuttals-3 | 8x H100 SXM 80GB |
| pod4 | thomas-rebuttals-4 | 8x H100 SXM 80GB |

## Git Access

All 4 pods have `~/.git-credentials` with GitHub PAT for `superkaiba`. `git pull/push` works without auth.

## Pod IP Changes

IPs change on restart. Update BOTH:
1. `.claude/mcp.json` → env vars for the pod
2. `~/.ssh/config` → HostName and Port for the alias
