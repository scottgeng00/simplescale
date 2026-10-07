from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

CPUS_PER_NODE = 192
MEMORY_MB_PER_NODE = 1_992_294


def _tp(value: str) -> int:
    tp = int(value)
    if tp < 1 or tp > 8 or tp & (tp - 1):
        raise argparse.ArgumentTypeError("TP must be one of 1, 2, 4, 8")
    return tp


def _pool(value: str) -> tuple[str, int]:
    qos, separator, count = value.partition("=")
    if not separator or not qos or not count.isdigit() or int(count) < 1:
        raise argparse.ArgumentTypeError("pool must be QOS=COUNT")
    return qos, int(count)


def _submit(command: list[str], dry_run: bool) -> str:
    if dry_run:
        print(shlex.join(command))
        return "dry-run"
    output = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()
    return output.split(";")[0]


def submit(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    manager_script, worker_script = repo / "slurm/manager.sbatch", repo / "slurm/worker.sbatch"
    if not manager_script.exists() or not worker_script.exists():
        raise SystemExit(f"Slurm scripts not found under {repo}")
    run = Path(args.run_dir).resolve()
    results, state, logs = run / "results", run / "state", run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manager_command = [
        "sbatch", "--parsable", f"--job-name={args.job_name}-manager",
        f"--chdir={repo}",
        f"--output={logs}/manager-%j.out", str(manager_script),
        str(repo / ".venv/bin/simplescale-manager"),
        "--manifest", str(Path(args.manifest).resolve()),
        "--output-dir", str(results), "--state-dir", str(state),
        "--chunk-size", str(args.chunk_size),
        "--start-workers", str(args.start_workers),
    ]
    if args.max_documents is not None:
        manager_command.extend(["--max-documents", str(args.max_documents)])
    if args.wandb_project:
        manager_command.extend(
            [
                "--wandb-project", args.wandb_project,
                "--wandb-name", args.wandb_name or run.name,
            ]
        )
    manager = _submit(manager_command, args.dry_run)
    print(f"manager {manager}")
    worker = [
        str(worker_script), str(repo / ".venv/bin/simplescale-worker"),
        "--discovery-file", str(state / "discovery.json"),
        "--handler", args.handler, "--model", str(Path(args.model).resolve()),
        "--tp-size", str(args.tp), "--context-length", str(args.context_length),
        "--task-concurrency", str(args.task_concurrency),
        "--startup-timeout", str(args.startup_timeout),
        "--startup-attempts", str(args.startup_attempts),
    ]
    if args.prefetch is not None:
        worker.extend(["--prefetch", str(args.prefetch)])
    cpus = CPUS_PER_NODE * args.tp // 8
    memory = MEMORY_MB_PER_NODE * args.tp // 8
    for qos, count in args.pool or [("h200_dream_high", 1)]:
        job = _submit(
            [
                "sbatch",
                "--parsable",
                f"--job-name={args.job_name}-{qos}",
                f"--qos={qos}",
                f"--array=0-{count - 1}",
                f"--gres=gpu:h200:{args.tp}",
                f"--cpus-per-task={cpus}",
                f"--mem={memory}M",
                f"--time={args.worker_time}",
                f"--chdir={repo}",
                f"--output={logs}/worker-{qos}-%A_%a.out",
                *worker,
            ],
            args.dry_run,
        )
        print(f"{qos} {job} ({count} workers, TP{args.tp})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simplescale")
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("submit", help="submit a manifest to Slurm")
    command.add_argument("manifest")
    command.add_argument("--model", required=True)
    command.add_argument("--handler", required=True)
    command.add_argument("--run-dir", required=True)
    command.add_argument("--pool", action="append", type=_pool, metavar="QOS=COUNT")
    command.add_argument("--tp", type=_tp, default=8)
    command.add_argument("--chunk-size", type=int, default=64)
    command.add_argument("--start-workers", type=int, default=1)
    command.add_argument("--max-documents", type=int)
    command.add_argument("--context-length", type=int, default=16384)
    command.add_argument("--task-concurrency", type=int, default=512)
    command.add_argument("--prefetch", type=int)
    command.add_argument("--startup-timeout", type=float, default=1200)
    command.add_argument("--startup-attempts", type=int, default=2)
    command.add_argument("--worker-time", default="2-00:00:00")
    command.add_argument("--wandb-project")
    command.add_argument("--wandb-name")
    command.add_argument("--job-name", default="simplescale")
    command.add_argument("--repo", default=".")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(run=submit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
