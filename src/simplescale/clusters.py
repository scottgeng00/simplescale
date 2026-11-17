"""Per-cluster Slurm settings, chosen automatically at submit time.

Detection: ``$SLURM_CLUSTER_NAME`` (else ``scontrol show config``'s ClusterName) is looked up by name; if the
name is unknown, the GPU type in the partitions' GRES (``sinfo``) picks the profile. ``simplescale submit
--cluster NAME`` overrides detection. To add a cluster, add a ``Cluster`` to ``CLUSTERS`` (README: "Setting up
on a new cluster").

Toolchain differences between clusters (Python headers, CUDA toolkit for SGLang/FlashInfer JIT) are detected by
``slurm/worker.sbatch`` on the node itself, so profiles only describe Slurm resources and default server flags.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cluster:
    name: str
    gpus_per_node: int
    cpus_per_node: int
    memory_mb_per_node: int  # requestable memory per full node; workers get tp / gpus_per_node of each
    gres: str  # GPU request for a TP-n worker, formatted with n
    default_pool: str  # QoS used when no --pool is given
    max_tp: int = 8
    account: str | None = None
    partition: str | None = None  # None: the site infers it (e.g. from the QoS)
    manager: tuple[str, ...] = ()  # sbatch flags for the CPU-only manager job
    server_args: tuple[str, ...] = ()  # default extra `sglang serve` flags for workers
    names: tuple[str, ...] = ()  # Slurm ClusterName values
    gpu_types: tuple[str, ...] = ()  # GRES GPU types identifying the cluster when its name is unknown
    notes: str = field(default="", compare=False)

    def worker_resources(self, tp: int) -> list[str]:
        if tp > min(self.max_tp, self.gpus_per_node):
            raise SystemExit(f"TP{tp} exceeds the {self.gpus_per_node} GPUs per node on {self.name}")
        return [
            *([f"--account={self.account}"] if self.account else []),
            *([f"--partition={self.partition}"] if self.partition else []),
            f"--gres={self.gres.format(n=tp)}",
            f"--cpus-per-task={self.cpus_per_node * tp // self.gpus_per_node}",
            f"--mem={self.memory_mb_per_node * tp // self.gpus_per_node}M",
        ]


CLUSTERS = {
    cluster.name: cluster
    for cluster in (
        Cluster(
            name="h200",
            gpus_per_node=8,
            cpus_per_node=192,
            memory_mb_per_node=1_992_294,
            gres="gpu:h200:{n}",
            default_pool="h200_dream_high",
            manager=("--partition=cpu", "--qos=cpu_lowest"),
            gpu_types=("h200",),
            notes="original H200 cluster; the partition is inferred from the QoS",
        ),
        Cluster(
            name="gb300-dalton",
            gpus_per_node=4,
            cpus_per_node=144,
            memory_mb_per_node=860_000,  # of 875,520 MB per node
            gres="gpu:{n}",
            default_pool="g3_comm_shared",
            max_tp=4,
            account="mnm",
            partition="g3",
            # No CPU partition for aarch64 binaries: the manager is a GPU-less job on g3.
            manager=("--account=mnm", "--partition=g3", "--qos=g3_comm_shared"),
            # FlashInfer's JIT sampling kernels do not build with the venv's CUDA 13 nvcc.
            server_args=("--sampling-backend pytorch",),
            names=("fair-cw-use2-3",),
            gpu_types=("gb300",),
            notes="FAIR CW USE2-3 (Dalton): aarch64, 4x GB300 per node",
        ),
        Cluster(
            name="h100-dm1",
            gpus_per_node=8,
            cpus_per_node=128,
            memory_mb_per_node=1_800_000,  # of 1,856,067 MB per node
            gres="gpu:h100:{n}",
            # h100_mdf_lowest is the preemptible shared pool (1,024-GPU cap); scale/explore preempt it.
            default_pool="h100_mdf_lowest",
            account="dream",
            partition="h100_mdf",
            # No CPU partition: the GPU-less manager runs on h100_mdf at a QoS that lowest jobs cannot preempt.
            manager=("--account=dream", "--partition=h100_mdf", "--qos=h100_mdf_scale"),
            names=("fair-cw-use1-1",),
            gpu_types=("h100",),
            notes="FAIR CW USE1-1 (DM1): x86_64, 8x H100 per node",
        ),
    )
}


def _cluster_name() -> str | None:
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    try:
        out = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ClusterName":
            return value.strip()
    return None


def _gpu_types() -> set[str]:
    try:
        out = subprocess.run(["sinfo", "-h", "-o", "%G"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    types = set()
    for gres in out.split():
        for item in gres.split(","):
            parts = item.split("(")[0].split(":")
            if parts[0] == "gpu" and len(parts) >= 3:
                types.add(parts[1].lower())
    return types


def detect(override: str | None = None) -> Cluster:
    """The profile for --cluster, else for this cluster's name, else for its GPU type."""
    if override:
        if override not in CLUSTERS:
            raise SystemExit(f"unknown --cluster {override!r}; known: {', '.join(CLUSTERS)}")
        return CLUSTERS[override]
    name = _cluster_name()
    for cluster in CLUSTERS.values():
        if name in cluster.names:
            return cluster
    types = _gpu_types()
    matches = [c for c in CLUSTERS.values() if types & set(c.gpu_types)]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"cannot pick a cluster profile (ClusterName={name!r}, GPU types={sorted(types)}); "
        f"pass --cluster ({', '.join(CLUSTERS)}) or add this cluster to simplescale/clusters.py"
    )
