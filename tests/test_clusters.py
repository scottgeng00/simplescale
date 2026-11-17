import pytest

from simplescale import clusters
from simplescale.cli import build_parser


def test_detect_by_name(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fair-cw-use1-1")
    assert clusters.detect().name == "h100-dm1"
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fair-cw-use2-3")
    assert clusters.detect().name == "gb300-dalton"


def test_detect_by_gpu_type(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "somewhere-else")
    monkeypatch.setattr(clusters, "_gpu_types", lambda: {"h200"})
    assert clusters.detect().name == "h200"
    monkeypatch.setattr(clusters, "_gpu_types", lambda: {"a100"})
    with pytest.raises(SystemExit):
        clusters.detect()
    assert clusters.detect("h100-dm1").name == "h100-dm1"


def test_worker_resources():
    assert clusters.CLUSTERS["gb300-dalton"].worker_resources(1) == [
        "--account=mnm", "--partition=g3", "--gres=gpu:1", "--cpus-per-task=36", "--mem=215000M"]
    assert clusters.CLUSTERS["h100-dm1"].worker_resources(2) == [
        "--account=dream", "--partition=h100_mdf", "--gres=gpu:h100:2", "--cpus-per-task=32", "--mem=450000M"]
    assert clusters.CLUSTERS["h200"].worker_resources(8) == ["--gres=gpu:h200:8", "--cpus-per-task=192",
                                                             "--mem=1992294M"]
    with pytest.raises(SystemExit):
        clusters.CLUSTERS["gb300-dalton"].worker_resources(8)


def test_submit_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fair-cw-use2-3")
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(["submit", "m.jsonl", "--model", "x", "--handler", "h", "--run-dir",
                                      str(tmp_path), "--tp", "1", "--repo", str(repo), "--dry-run"])
    args.run(args)
    out = capsys.readouterr().out.splitlines()
    manager = next(line for line in out if "-manager" in line)
    worker = next(line for line in out if "--array" in line)
    assert "--partition=g3 --qos=g3_comm_shared" in manager
    assert "--qos=g3_comm_shared" in worker and "--gres=gpu:1" in worker
    assert "'--server-arg=--sampling-backend pytorch'" in worker


def test_prefetch_follows_pool(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fair-cw-use1-1")
    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(["submit", "m.jsonl", "--model", "x", "--handler", "h", "--run-dir",
                                      str(tmp_path), "--tp", "1", "--repo", str(repo), "--dry-run",
                                      "--pool", "h100_mdf_explore=2", "--pool", "h100_mdf_lowest=4"])
    args.run(args)
    workers = [line for line in capsys.readouterr().out.splitlines() if "--array" in line]
    assert "--qos=h100_mdf_explore" in workers[0] and workers[0].endswith("--prefetch 8")
    assert "--qos=h100_mdf_lowest" in workers[1] and workers[1].endswith("--prefetch 6")
    assert "--gres=gpu:h100:1" in workers[1] and "--partition=h100_mdf" in workers[1]


def test_pools():
    pools = {name: (c.high_pool, c.bulk_pool) for name, c in clusters.CLUSTERS.items()}
    assert pools == {"h200": ("h200_dream_high", "h200_comm_shared"),
                     "gb300-dalton": ("g3_mnm_high", "g3_comm_shared"),
                     "h100-dm1": ("h100_mdf_explore", "h100_mdf_lowest")}
