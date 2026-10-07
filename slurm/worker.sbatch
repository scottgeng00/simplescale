#!/usr/bin/env bash
#SBATCH --job-name=simplescale-worker
#SBATCH --partition=h200
#SBATCH --qos=h200_dream_high
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1992294M
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cache_root="${SLURM_TMPDIR:-/tmp}/simplescale-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
export TRITON_CACHE_DIR="${cache_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

drain_worker() {
  local requeue="${1}"
  trap '' USR1 TERM
  if [[ -n "${worker_pid:-}" ]]; then
    kill -USR1 "${worker_pid}" 2>/dev/null || true
    wait "${worker_pid}" 2>/dev/null || true
  fi
  if [[ "${requeue}" == "1" ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
  fi
  exit 0
}
trap 'drain_worker 1' USR1
trap 'drain_worker 0' TERM

"$@" &
worker_pid=$!
set +e
wait "${worker_pid}"
code=$?
set -e
if (( code != 0 && ${SLURM_RESTART_COUNT:-0} < 10 )); then
  echo "worker exited ${code}; requeueing (${SLURM_RESTART_COUNT:-0}/10)" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${code}"
