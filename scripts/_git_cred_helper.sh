#!/bin/bash
# Source this file to define the pod git-auth constants (#1239, #1401).
# Requires: $REMOTE_DIR set to the pod repo dir (the helper's .env-fallback
# path is baked in at expansion time). Consumers: bootstrap_pod.sh,
# sync_env_keys.sh. Single source of truth — do NOT redefine these
# elsewhere (drift risk; quoting invariant pinned by
# tests/test_bootstrap_pod_git_credentials.py).

if [ -z "${REMOTE_DIR:-}" ]; then
    echo "ERROR: _git_cred_helper.sh requires \$REMOTE_DIR to be set before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi

# Tokenless public HTTPS remote (#1239, mirrors gcp.py DEFAULT_REPO_URL —
# the repo is public, so CLONE/FETCH needs no auth; PUSH auth comes from
# the credential helper below).
REPO_URL_TOKENLESS="https://github.com/superkaiba/explore-persona-space.git"

# Env-reading git credential helper (#1205 GCE-parity, pod flavor). This
# STRING is what gets stored in git config — never the token. Git runs it
# via `sh -c` with the credential operation appended ("get"/"store"/
# "erase" — f ignores it, same as the GCE helper). It reads GITHUB_TOKEN
# from the invoking environment first, then falls back to sourcing the
# durable pod .env in a subshell (a later interactive/SSH shell on the
# pod does NOT inherit the bootstrap env — the KEY delta vs GCE). It is
# configured host-scoped (credential.https://github.com.helper) so the
# token is never offered to any non-GitHub remote. POSIX-sh only (git
# invokes /bin/sh, dash on this image). Escaping note: this is a LOCAL
# double-quoted assignment — \" and \$ survive as literal " and $ in the
# stored value; $REMOTE_DIR is expanded (baked in) locally. The value
# contains NO single quotes by construction — every use site interpolates
# it inside remote-level single quotes (quoting invariant, pinned by
# tests/test_bootstrap_pod_git_credentials.py).
# Named deviation vs #1205 (plan §11): the pod installs the helper
# UNCONDITIONALLY where GCE gates on token presence — justified by the
# retained step-3 GITHUB_TOKEN-in-.env gate plus the empty-password
# fail-loud degrade when the token is genuinely absent at invocation.
GIT_CRED_HELPER="!f() { tok=\"\${GITHUB_TOKEN:-}\"; if [ -z \"\$tok\" ] && [ -r $REMOTE_DIR/.env ]; then tok=\"\$(. $REMOTE_DIR/.env >/dev/null 2>&1; printf %s \"\${GITHUB_TOKEN:-}\")\"; fi; echo username=x-access-token; echo \"password=\${tok}\"; }; f"
