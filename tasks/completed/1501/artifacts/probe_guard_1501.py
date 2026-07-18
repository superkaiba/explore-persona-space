import json
import subprocess

GUARD = "/home/thomasjiralerspong/explore-persona-space/scripts/guard_repo_root_branch.sh"

MERGE = "git " + "merge"  # avoid embedding gated literals in THIS file's launch command
GHMERGE = "gh pr " + "merge"
BT = chr(96)


def probe(name, command):
    payload = json.dumps({"tool_input": {"command": command}})
    r = subprocess.run(["bash", GUARD], input=payload, capture_output=True, text=True)
    verdict = "ALLOW" if r.returncode == 0 else f"BLOCK(rc={r.returncode})"
    print(f"{name}: {verdict}")


# A: quoted tag, plain body mentioning merge verb
probe(
    "A quoted-tag plain body     ",
    f"cat > /tmp/x.md <<'EOF'\nrun {GHMERGE} --rebase to land it\nEOF",
)
# B: UNQUOTED tag, plain body (no backticks/$) mentioning merge verb
probe(
    "B unquoted-tag plain body   ", f"cat > /tmp/x.md <<EOF\nrun {GHMERGE} --rebase to land it\nEOF"
)
# C: UNQUOTED tag, body line with BACKTICKS around the verb (markdown style)
probe(
    "C unquoted-tag backticks    ", f"cat > /tmp/x.md <<EOF\nrun {BT}{GHMERGE}{BT} to land it\nEOF"
)
# D: QUOTED tag, body with backticks (markdown style)
probe(
    "D quoted-tag backticks      ",
    f"cat > /tmp/x.md <<'EOF'\nrun {BT}{GHMERGE}{BT} to land it\nEOF",
)
# E: UNQUOTED tag, body with dollar-brace + verbatim git-merge line
probe(
    "E unquoted-tag dollar-brace ",
    f"cat > /tmp/x.md <<EOF\nthe fix: {MERGE} origin/main at ${{REPO}}\nEOF",
)
# F: quoted tag, body mentioning subprocess (check f) + merge verb
probe(
    "F quoted-tag subprocess     ",
    f"cat > /tmp/x.md <<'EOF'\nthe helper calls subprocess.run and then {MERGE} lands\nEOF",
)
# G: compound heredoc + trailing benign command (the /daily filing shape)
probe(
    "G compound quoted-tag       ",
    f"cat > /tmp/wf.md <<'EOF'\n## gap\nnever run a bare {MERGE} at the repo root\nEOF\nuv run python scripts/task.py view 1501",
)
# H: unterminated heredoc (pass-1 refusal) with merge mention
probe("H unterminated heredoc      ", f"cat > /tmp/x.md <<'EOF'\nnever {MERGE} at root")
# I: bash consumer with gated verb in body (must stay BLOCKED)
probe("I bash consumer (must block)", f"bash <<'EOF'\n{MERGE} origin/main\nEOF")
# J: quoted tag body with $( expansion syntax (check g skipped for quoted)
probe("J quoted-tag cmdsub in body ", f"cat > /tmp/x.md <<'EOF'\nuse $(date) then {MERGE} it\nEOF")
# K: unquoted tag, cmdsub in body (check g refusal) + merge mention
probe("K unquoted-tag cmdsub       ", f"cat > /tmp/x.md <<EOF\nuse $(date) then {MERGE} it\nEOF")
# L: two heredocs in one compound, both quoted, merge mention in second
probe(
    "L two quoted heredocs       ",
    f"cat > /tmp/a.md <<'EOF'\nhello\nEOF\ncat > /tmp/b.md <<'EOF'\nthe {MERGE} fence\nEOF",
)
# M: tee consumer, quoted tag
probe(
    "M tee quoted-tag            ", f"tee /tmp/x.md > /dev/null <<'EOF'\nnever {MERGE} at root\nEOF"
)
# N: python heredoc consumer (not in shellish list) with subprocess+verb (real risk shape)
probe(
    "N python consumer subprocess",
    "uv run python - <<'EOF'\nimport subprocess\nsubprocess.run(['git','checkout','-b','x'])\nEOF",
)
