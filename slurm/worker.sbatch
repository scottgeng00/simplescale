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
wait "${worker_pid}"
