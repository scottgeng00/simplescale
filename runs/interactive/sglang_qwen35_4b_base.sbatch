#!/bin/bash
#SBATCH --job-name=sglang-qwen35-4b-base
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --chdir=/home/sgeng/scottg/pretrain/simplescale
# Interactive Qwen3.5-4B-Base sglang server; writes its address to runs/interactive/sglang_qwen35_4b_base.addr.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cache_root="${SLURM_TMPDIR:-/tmp}/sglang-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${cache_root}/triton" TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer" SGLANG_JIT_CACHE_DIR="${cache_root}/sglang_jit"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"
# Toolchain as in slurm/worker.sbatch: Python.h from uv's CPython, nvcc / libcudart from the venv's CUDA wheel.
python_include="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [[ ! -f "${python_include}/Python.h" ]]; then
  version="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  for include in "${HOME}"/.local/share/uv/python/cpython-"${version}".*-"$(uname -m)"-*/include/python"${version}"; do
    [[ -f "${include}/Python.h" ]] && export CPATH="${include}${CPATH:+:${CPATH}}" && break
  done
fi
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
# Auto-requeue: Slurm requeues on preemption (--requeue); near the time limit USR1 requeues the job itself; if the
# server process dies (e.g. OOM) it is restarted in place. The address file is written only once /health answers.
trap 'echo "$(date) time limit near: requeue"; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
while true; do
  PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")  # free port: nodes are shared
  .venv/bin/sglang serve --model-path /home/sgeng/scottg/pretrain/assets/models/Qwen3.5-4B-Base \
  --host 0.0.0.0 --port ${PORT} --context-length 32768 --mem-fraction-static 0.75 &
  SERVER=$!
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || ! kill -0 "${SERVER}" 2>/dev/null; do sleep 10; done
  kill -0 "${SERVER}" 2>/dev/null && echo "http://$(hostname -f):${PORT}" > runs/interactive/sglang_qwen35_4b_base.addr
  wait "${SERVER}" || true
  echo "$(date) sglang exited; restarting in 30 s"
  sleep 30
done
