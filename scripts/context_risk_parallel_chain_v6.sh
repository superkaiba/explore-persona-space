#!/usr/bin/env bash
# Join the two parallel context-risk stages and run the prospective analysis.
#
#   stage A  misalignment rollouts on pod-2 (launched independently, see
#            <remote root>/launch_provenance.json)
#   stage B  prompt-B reward-hacking rerun on pod-0 (systemd user unit on this VM)
#
# Each poll checks both stages. When stage A completes it is copied back,
# verified, and pod-2 (provisioned by this chain's author) is terminated. When
# stage B completes the model server on pod-0 is stopped; pod-0 itself is only
# terminated when EPM_CONTEXT_RISK_TERMINATE_POD0=1 (pod-0 was not provisioned
# by this chain, so that needs Thomas's word). When both are done the analysis
# runs. Status is written to a JSON file after every transition.
#
# Stage B completion gate: 480 realized rollouts, 0 technical errors, prompt B.
# A pass that ends with errored pairs (e.g. a CUDA out-of-memory 500 on a long
# prefill) or an evaluator crash gets a lossless resume pass through the launch
# script's `resume` mode (completed pairs reused, the rest re-run), at most
# EPM_CONTEXT_RISK_MAX_RESUMES times, before the stage is declared failed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi

pod2_host="${EPM_CONTEXT_RISK_POD2_HOST:-103.207.149.173}"
pod2_port="${EPM_CONTEXT_RISK_POD2_PORT:-15150}"
pod2_id="${EPM_CONTEXT_RISK_POD2_ID:-9avg3ujei1p9s3}"
pod0_host="${EPM_CONTEXT_RISK_POD0_HOST:-64.247.201.59}"
pod0_port="${EPM_CONTEXT_RISK_POD0_PORT:-13363}"
pod0_id="${EPM_CONTEXT_RISK_POD0_ID:-agq359n2pvzlb6}"
pod0_server_root="${EPM_CONTEXT_RISK_POD0_SERVER_ROOT:-/workspace/context_risk_results/qwen38_batched_server_v19}"
terminate_pod0="${EPM_CONTEXT_RISK_TERMINATE_POD0:-0}"
mis_remote="${EPM_CONTEXT_RISK_MIS_REMOTE:-/workspace/context_risk_results/qwen38_misalignment_public_v5}"
mis_local="${EPM_CONTEXT_RISK_MIS_LOCAL:-eval_results/context_risk/qwen38_misalignment_public_v5}"
mis_expected_rollouts="${EPM_CONTEXT_RISK_MIS_EXPECTED:-640}"
reward_root="${EPM_CONTEXT_RISK_REWARD_ROOT:-eval_results/context_risk/impossible_livecodebench_v19}"
reward_unit="${EPM_CONTEXT_RISK_REWARD_UNIT:-context-risk-reward-v19-full.service}"
resume_unit="${EPM_CONTEXT_RISK_RESUME_UNIT:-context-risk-reward-v19-resume.service}"
reward_launcher="${EPM_CONTEXT_RISK_REWARD_LAUNCHER:-scripts/context_risk_promptB_reward_launch.sh}"
max_resumes="${EPM_CONTEXT_RISK_MAX_RESUMES:-3}"
resume_counter="$reward_root/resume_attempts"
tunnel_unit="${EPM_CONTEXT_RISK_TUNNEL_UNIT:-context-risk-v18-recovery2-tunnel.service}"
capture_root="${EPM_CONTEXT_RISK_CAPTURE_ROOT:-eval_results/context_risk/qwen38_impossible_contexts_v4_promptB}"
impossible_manifest="${EPM_CONTEXT_RISK_IMPOSSIBLE_MANIFEST:-eval_results/context_risk/data/impossible_livecodebench_promptB/public_pilot_manifest.jsonl}"
analysis_root="${EPM_CONTEXT_RISK_ANALYSIS_OUT:-eval_results/context_risk/prospective_analysis_v3}"
status_path="${EPM_CONTEXT_RISK_CHAIN_STATUS:-eval_results/context_risk/parallel_chain_v5_status.json}"
poll_seconds="${EPM_CONTEXT_RISK_POLL_SECONDS:-300}"

ssh_common=(-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
ssh2() { ssh "${ssh_common[@]}" -p "$pod2_port" "root@$pod2_host" "$@"; }
ssh0() { ssh "${ssh_common[@]}" -p "$pod0_port" "root@$pod0_host" "$@"; }

mis_state="running"; reward_state="running"; phase="waiting"; needs_you=""
mis_note=""; reward_note=""
write_status() {
  local state="$1"
  mkdir -p "$(dirname "$status_path")"
  python3 - "$status_path" "$state" "$phase" "$mis_state" "$mis_note" "$reward_state" "$reward_note" "$needs_you" <<'PY'
import datetime, json, sys, tempfile, os
path, state, phase, ms, mn, rs, rn, ny = sys.argv[1:]
payload = {"schema_version": "context_risk_parallel_chain_v6_status", "state": state, "phase": phase,
           "misalignment": {"state": ms, "note": mn}, "reward": {"state": rs, "note": rn},
           "needs_you": ny, "updated_at": datetime.datetime.now().astimezone().isoformat()}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
with os.fdopen(fd, "w") as fh: json.dump(payload, fh, indent=2, sort_keys=True)
os.replace(tmp, path)
PY
}
log() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }
terminate_pod() {  # $1 = pod id
  [[ -n "${RUNPOD_API_KEY:-}" ]] || { log "RUNPOD_API_KEY missing; cannot terminate $1"; return 1; }
  local resp
  resp="$(curl -sS -m 60 -X POST -H 'content-type: application/json' \
    "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$1\\\"}) }\"}")"
  if printf '%s' "$resp" | grep -q '"errors"'; then log "terminate $1 failed: $resp"; return 1; fi
  log "terminated pod $1"
}

trap 'phase="interrupted"; write_status "failed"' TERM INT
write_status "running"

check_misalignment() {
  local remote_state
  remote_state="$(ssh2 "
    cd '$mis_remote' 2>/dev/null || { printf missing; exit 0; }
    pid=\$(cat queue.pid 2>/dev/null || echo 0)
    if test -d /proc/\$pid; then printf alive; exit 0; fi
    if test -s run_result.json; then
      python3 -c 'import json,sys; r=json.load(open(\"run_result.json\")); ok=r.get(\"passed\") and r.get(\"n_rollouts\")==int(sys.argv[1]); print(\"complete\" if ok else \"badresult\", r.get(\"n_rollouts\"), r.get(\"prevalence_gate_passed\"), r.get(\"censoring_gate_passed\"), r.get(\"n_positive\"), r.get(\"n_negative\"), r.get(\"n_censored\"))' '$mis_expected_rollouts'
    else printf dead; fi")" || { log "pod-2 ssh failed"; return 0; }
  case "$remote_state" in
    alive|missing) return 0 ;;
    dead) mis_state="failed"; mis_note="rollout process exited without run_result.json (see $mis_remote/queue.log on pod-2)"; needs_you="misalignment failed on pod-2; pod-2 left running for inspection ($pod2_id, \$3.49/h)"; write_status "running"; return 0 ;;
    badresult*) mis_state="failed"; mis_note="run_result.json present but gate failed: $remote_state"; needs_you="misalignment result failed its completion gate; pod-2 left running ($pod2_id)"; write_status "running"; return 0 ;;
    complete*) ;;
    *) log "unexpected pod-2 state: $remote_state"; return 0 ;;
  esac
  phase="copying_misalignment"; mis_note="$remote_state"; write_status "running"
  mkdir -p "$mis_local"
  ssh2 "tar -C '$mis_remote' -cf - ." | tar -C "$mis_local" -xf -
  local remote_sha local_sha
  remote_sha="$(ssh2 "sha256sum '$mis_remote/run_result.json' | cut -d' ' -f1")"
  local_sha="$(sha256sum "$mis_local/run_result.json" | cut -d' ' -f1)"
  if [[ "$remote_sha" != "$local_sha" ]]; then mis_note="copy verification failed"; write_status "running"; return 0; fi
  local n_ctx; n_ctx="$(find "$mis_local" -maxdepth 1 -type d -name 'context_*' | wc -l)"
  mis_state="done"; mis_note="copied $n_ctx context dirs; $remote_state"
  if terminate_pod "$pod2_id"; then mis_note="$mis_note; pod-2 terminated"; else needs_you="${needs_you:+$needs_you; }pod-2 $pod2_id still running: terminate it"; fi
  phase="waiting"; write_status "running"
}

launch_resume() {  # $1 = reason; returns 0 when a lossless resume pass was launched
  local n; n="$(cat "$resume_counter" 2>/dev/null || echo 0)"
  if (( n >= max_resumes )); then log "resume budget exhausted ($n/$max_resumes): $1"; return 1; fi
  if "$reward_launcher" resume >>"$reward_root/resume_launch.log" 2>&1; then
    echo "$((n + 1))" > "$resume_counter"
    reward_note="lossless resume $((n + 1))/$max_resumes launched: $1"
    log "$reward_note"; write_status "running"; return 0
  fi
  log "resume launch failed ($1); see $reward_root/resume_launch.log"; return 1
}

check_reward() {
  local result="$reward_root/full/run_result.json"
  # While either evaluator unit runs it owns <reward_root>/full; any run_result.json
  # on disk then belongs to an earlier pass and is not judged.
  if systemctl --user is-active --quiet "$reward_unit" || systemctl --user is-active --quiet "$resume_unit"; then return 0; fi
  if [[ ! -s "$result" ]]; then
    launch_resume "evaluator unit exited without run_result.json" && return 0
    reward_state="failed"; reward_note="evaluator exited without run_result.json and no resume could be launched"
    needs_you="${needs_you:+$needs_you; }prompt-B reward run died (see $reward_root/full.log, resume.log, resume_launch.log); server on pod-0 left up"; write_status "running"; return 0
  fi
  local verdict
  verdict="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("ok" if r["passed"] and r["realized_rollouts"]==480 and r["technical_errors"]==0 and r.get("prompt_variant")=="B" else "bad", r["realized_rollouts"], r["technical_errors"], r.get("prompt_variant"), r.get("reward_hacking_prevalence_gate",{}).get("passed"))' "$result")"
  reward_note="$verdict"
  if [[ "$verdict" != ok* ]]; then
    local realized errors variant
    read -r _ realized errors variant _ <<<"$verdict"
    if [[ "$variant" == B ]] && (( errors > 0 || realized < 480 )); then
      launch_resume "run_result realized=$realized technical_errors=$errors" && return 0
    fi
    reward_state="failed"; needs_you="${needs_you:+$needs_you; }prompt-B reward run_result failed its gate: $verdict (resumes used: $(cat "$resume_counter" 2>/dev/null || echo 0)/$max_resumes)"; write_status "running"; return 0
  fi
  phase="stopping_pod0_server"; write_status "running"
  ssh0 "pid=\$(cat '$pod0_server_root/server.pid' 2>/dev/null || echo); if [ -n \"\$pid\" ] && test -d /proc/\$pid; then kill -TERM -- -\$pid; for i in \$(seq 1 120); do test -d /proc/\$pid || exit 0; sleep 1; done; kill -KILL -- -\$pid; fi" || log "server stop on pod-0 failed"
  systemctl --user stop "$tunnel_unit" || true
  reward_state="done"
  if [[ "$terminate_pod0" == 1 ]]; then terminate_pod "$pod0_id" && reward_note="$reward_note; pod-0 terminated"; else needs_you="${needs_you:+$needs_you; }pod-0 $pod0_id is idle (\$3.29/h): say the word to terminate it"; fi
  phase="waiting"; write_status "running"
}

while true; do
  [[ "$mis_state" == running ]] && check_misalignment
  [[ "$reward_state" == running ]] && check_reward
  if [[ "$mis_state" == done && "$reward_state" == done ]]; then break; fi
  if [[ "$mis_state" == failed && "$reward_state" != running ]] || [[ "$reward_state" == failed && "$mis_state" != running ]]; then
    phase="blocked"; write_status "failed"; exit 4
  fi
  sleep "$poll_seconds"
done

phase="prospective_analysis"; write_status "running"
uv run --with scikit-learn python scripts/context_risk_analyze.py \
  --impossible-result "$reward_root/full/run_result.json" \
  --impossible-capture-root "$capture_root" \
  --impossible-manifest "$impossible_manifest" \
  --map-artifact eval_results/context_risk/qwen38_map_pilot/map_layer_44.npz \
  --misalignment-result "$mis_local/run_result.json" \
  --misalignment-rollout-root "$mis_local" \
  --misalignment-manifest eval_results/context_risk/data/agentic_misalignment/public_development_manifest.jsonl \
  --output-dir "$analysis_root"
phase="complete"; write_status "completed"
trap - TERM INT
