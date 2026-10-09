from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a JSONL pointer manifest")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-document-tokens", type=int)
    parser.add_argument(
        "--sampling-params", type=json.loads, default={"temperature": 1.0}
    )
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")
    if args.max_document_tokens is not None and args.max_document_tokens < 1:
        parser.error("--max-document-tokens must be positive")
    if not isinstance(args.sampling_params, dict):
        parser.error("--sampling-params must be a JSON object")

    source, output = args.source.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    offset = 0
    with source.open("rb") as raw, temporary.open("w") as manifest:
        for _ in range(args.rows):
            line = raw.readline()
            if not line:
                raise ValueError(f"{source} has fewer than {args.rows} rows")
            task = {
                "_ref": {"path": str(source), "offset": offset, "length": len(line)},
                "generations": args.generations,
                "sampling_params": args.sampling_params,
            }
            if args.max_document_tokens is not None:
                task["max_document_tokens"] = args.max_document_tokens
            manifest.write(json.dumps(task, separators=(",", ":")) + "\n")
            offset += len(line)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
