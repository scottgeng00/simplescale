#!/usr/bin/env bash
#SBATCH --job-name=simplescale-worker
# Account / partition / QoS / GPUs / CPUs / memory come from the cluster profile (simplescale/clusters.py).
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
# Toolchain for the Triton / SGLang / FlashInfer JIT, detected on the node (jobs run with --chdir=<repo>):
# - Triton compiles against Python.h; a venv on a header-less system Python borrows a uv CPython's headers.
python_include="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [[ ! -f "${python_include}/Python.h" ]]; then
  version="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  for include in "${HOME}"/.local/share/uv/python/cpython-"${version}".*-"$(uname -m)"-*/include/python"${version}"; do
    [[ -f "${include}/Python.h" ]] && export CPATH="${include}${CPATH:+:${CPATH}}" && break
  done
fi
# - Without a system CUDA toolkit, use nvcc / headers / libcudart from the venv's nvidia wheels (+ ninja from the venv).
#   The wheel ships only libcudart.so.<major>; a per-job directory provides libcudart.so for -lcudart.
export PATH="${PWD}/.venv/bin:${PATH}"
if ! command -v nvcc >/dev/null; then
  for cuda in .venv/lib/python3*/site-packages/nvidia/cu1[0-9]; do
    if [[ -x "${cuda}/bin/nvcc" ]]; then
      export CUDA_HOME="${PWD}/${cuda}"
      mkdir -p "${cache_root}/cuda_link"
      ln -sf "$(ls "${CUDA_HOME}"/lib/libcudart.so.* | head -1)" "${cache_root}/cuda_link/libcudart.so"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LIBRARY_PATH="${cache_root}/cuda_link:${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi
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
