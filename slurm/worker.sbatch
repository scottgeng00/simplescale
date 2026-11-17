#!/usr/bin/env bash
#SBATCH --job-name=simplescale-worker
#SBATCH --account=mnm
#SBATCH --partition=g3
#SBATCH --qos=g3_comm_shared
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --mem=860000M
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
export SGLANG_JIT_CACHE_DIR="${cache_root}/sglang_jit"  # the shared 0.5.21 JIT cache never hits (README)
# The venv runs on the system python3.12, which has no dev headers; Triton JIT needs Python.h.
export CPATH="/home/sgeng/.local/share/uv/python/cpython-3.12.15-linux-aarch64-gnu/include/python3.12${CPATH:+:${CPATH}}"
# SGLang JIT kernels build with ninja + nvcc from the venv (no system CUDA toolkit; jobs run with --chdir=<repo>).
export CUDA_HOME="${PWD}/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="${PWD}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
# The wheel ships only libcudart.so.13; .venv/cuda_link/libcudart.so links to it for -lcudart.
export LIBRARY_PATH="${PWD}/.venv/cuda_link:${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
