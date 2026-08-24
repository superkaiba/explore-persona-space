#!/bin/bash
# task #2378 recovery leg A+B (post-r14 relaunch; pod-2378-c).
# Leg A: SegA topups for the 3 G2b-shortfall cells (per-cell sizing from the
#        realized wave-1 rates x1.2 — epm:progress v82 basis), then direct
#        upload_stage for sega + sega_mined (the per-invocation Runner upload
#        steps resume-skip on identical argv across invocations, so the
#        launcher owns the final full-dir uploads).
# Leg B: plain regen under the r13/r14 think-ban (chat resumes via pre-staged
#        w1 ledgers at zero GPU; defaults: chat 12000 / plain 10000 rows).
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
echo $$ > /workspace/logs/issue-2378.pid
exec >> /workspace/logs/issue-2378-recovery.log 2>&1
echo "[phase=recovery_start] $(date -u +%FT%TZ)"
bash scripts/issue2378_dispatch.sh p4_topup --cells storyq_helios --sega-attempts-per-cell 500 || { echo "[phase=recovery_fail] topup helios rc=$?"; exit 10; }
bash scripts/issue2378_dispatch.sh p4_topup --cells storyq_vex --sega-attempts-per-cell 5000 || { echo "[phase=recovery_fail] topup vex rc=$?"; exit 11; }
bash scripts/issue2378_dispatch.sh p4_topup --cells storyq_dana --sega-attempts-per-cell 17500 || { echo "[phase=recovery_fail] topup dana rc=$?"; exit 12; }
uv run python scripts/issue2378_gen.py --phase upload_stage --stage sega || { echo "[phase=recovery_fail] upload sega rc=$?"; exit 13; }
uv run python scripts/issue2378_gen.py --phase upload_stage --stage sega_mined || { echo "[phase=recovery_fail] upload sega_mined rc=$?"; exit 14; }
# Leg B: 4-way shard fanout replicating the dispatcher's launcher-env pins
# (CVD in the LAUNCHER env — #545/#523 — plus cm.LAUNCH_ENV_PINS; shard count
# MUST be 4 to match the pre-staged chat w1 ledgers' regime shard [s, 4]).
pids=()
for s in 0 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$s VLLM_USE_FLASHINFER_SAMPLER=0 \
    /root/eps-model-venv/bin/python scripts/issue2378_gen.py --phase chat_plain \
    --stage-pools-from-hf --skip-upload --shard-index $s --num-shards 4 \
    >> /workspace/logs/issue-2378-recovery-chatplain-s$s.log 2>&1 &
  pids+=($!)
done
rcB=0
for p in "${pids[@]}"; do wait "$p" || rcB=$?; done
[ "$rcB" -eq 0 ] || { echo "[phase=recovery_fail] chat_plain rc=$rcB"; exit 15; }
uv run python scripts/issue2378_gen.py --phase upload_stage --stage plain || { echo "[phase=recovery_fail] upload plain rc=$?"; exit 16; }
uv run python scripts/issue2378_gen.py --phase upload_stage --stage chat || { echo "[phase=recovery_fail] upload chat rc=$?"; exit 17; }
echo "[phase=recovery_done] $(date -u +%FT%TZ)"
