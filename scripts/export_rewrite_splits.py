#!/usr/bin/env python3
"""Convert aligned SimpleScale rephrases to pretraining JSONL chunks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator


def rewrite_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or Path(name).name != name:
        raise argparse.ArgumentTypeError("rewrite must be NAME=DATASET_JSON")
    return name, Path(path)


def load_dataset(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    shards = metadata["shards"]
    if [shard["chunk"] for shard in shards] != list(range(len(shards))):
        raise ValueError(f"shards are not ordered by chunk in {path}")
    if len(shards) != metadata["chunks"] or sum(s["rows"] for s in shards) != metadata["rows"]:
        raise ValueError(f"inconsistent dataset metadata in {path}")
    metadata["dataset_path"] = str(path.resolve())
    return metadata


def iter_rows(metadata: dict) -> Iterator[dict]:
    for shard in metadata["shards"]:
        rows = 0
        with Path(shard["uri"]).open() as source:
            for line in source:
                rows += 1
                yield json.loads(line)
        if rows != shard["rows"]:
            raise ValueError(f"expected {shard['rows']} rows in {shard['uri']}, found {rows}")


def export(
    source: Path,
    source_name: str,
    rewrites: list[tuple[str, Path]],
    output: Path,
    num_chunks: int,
) -> None:
    datasets = [(name, load_dataset(path.resolve())) for name, path in rewrites]
    if not datasets:
        raise ValueError("at least one rewrite dataset is required")
    alignment = ("manifest", "sha256", "rows", "chunks", "chunk_size")
    expected = tuple(datasets[0][1][key] for key in alignment)
    if any(tuple(dataset[key] for key in alignment) != expected for _, dataset in datasets[1:]):
        raise ValueError("rewrite datasets were not generated from the same ordered manifest")

    rows = datasets[0][1]["rows"]
    if not 1 <= num_chunks <= rows:
        raise ValueError("num_chunks must be between 1 and the number of rows")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    iterators = [iter(iter_rows(dataset)) for _, dataset in datasets]
    names = [source_name, *(name for name, _ in datasets)]
    width = max(2, len(str(num_chunks - 1)))
    try:
        temporary.mkdir()
        for name in names:
            (temporary / name).mkdir()
        written = 0
        with ExitStack() as stack:
            source_file = stack.enter_context(source.resolve().open("rb"))
            files = [
                [
                    stack.enter_context(
                        (
                            temporary
                            / name
                            / f"{name}.chunk.{chunk:0{width}d}.jsonl"
                        ).open("wb", buffering=8 << 20)
                    )
                    for chunk in range(num_chunks)
                ]
                for name in names
            ]
            while written < rows:
                buffers = [
                    [bytearray() for _ in range(num_chunks)] for _ in names
                ]
                for _ in range(min(8192, rows - written)):
                    source_line = source_file.readline()
                    if not source_line:
                        raise ValueError(f"source contains fewer than {rows} rows")
                    records = [next(iterator) for iterator in iterators]
                    task_id = records[0]["task_id"]
                    if any(record["task_id"] != task_id for record in records[1:]):
                        raise ValueError(f"rewrite alignment mismatch at row {written}")
                    source_id = json.loads(source_line).get("id", written)
                    if source_id != task_id:
                        raise ValueError(f"source alignment mismatch at row {written}")
                    chunk = written % num_chunks
                    buffers[0][chunk].extend(
                        source_line if source_line.endswith(b"\n") else source_line + b"\n"
                    )
                    for dataset, record in enumerate(records, 1):
                        rephrases = record["result"]["rephrases"]
                        if len(rephrases) != 1:
                            raise ValueError(f"expected one rephrase at row {written}")
                        buffers[dataset][chunk].extend(
                            json.dumps(
                                {"id": task_id, "text": rephrases[0]},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode()
                            + b"\n"
                        )
                    written += 1
                for dataset in range(len(names)):
                    for chunk in range(num_chunks):
                        files[dataset][chunk].write(buffers[dataset][chunk])
                if written == rows or written % 1_000_000 < 8192:
                    print(f"wrote {written}/{rows} aligned rows", flush=True)
            if source_file.readline():
                raise ValueError(f"source contains more than {rows} rows")

        for iterator in iterators:
            try:
                next(iterator)
            except StopIteration:
                continue
            raise ValueError("rewrite dataset contains more rows than declared")
        (temporary / "dataset.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "chunks": num_chunks,
                    "distribution": "round_robin",
                    "manifest": datasets[0][1]["manifest"],
                    "manifest_sha256": datasets[0][1]["sha256"],
                    "datasets": {
                        source_name: str(source.resolve()),
                        **{
                            name: dataset["dataset_path"]
                            for name, dataset in datasets
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-chunks", type=int, default=16)
    parser.add_argument("--source", type=Path, default=repo / "runs/data/dclm.first-15m.jsonl")
    parser.add_argument("--source-name", default="dclm_15m")
    parser.add_argument(
        "--output-dir", type=Path, default=repo.parent / "assets/data/rewrite_splits"
    )
    parser.add_argument("--rewrite", action="append", type=rewrite_arg)
    args = parser.parse_args()
    rewrites = args.rewrite or [
        (
            "qwen35_2b_rephrase",
            repo / "runs/qwen35-2b-rephrase-g1-15m/results/dataset.json",
        ),
        (
            "qwen35_27b_rephrase",
            repo / "runs/qwen35-27b-rephrase-g1-15m/results/dataset.json",
        ),
    ]
    export(
        args.source,
        args.source_name,
        rewrites,
        args.output_dir.resolve(),
        args.num_chunks,
    )


if __name__ == "__main__":
    main()
