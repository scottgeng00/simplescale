#!/usr/bin/env bash
#SBATCH --job-name=simplescale-manager
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6-23:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

requeue_manager() {
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_manager USR1

"$@" &
manager_pid=$!
wait "${manager_pid}"
