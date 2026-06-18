#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from result_utils import GRID_FIELDS, default_root, metric_grid_rows, require_methods, write_csv_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", default="monim")
    parser.add_argument("--ckpt-name", default="gpt2-small")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir is not None else root / "output"
    rows, problems = metric_grid_rows(
        output_dir=output_dir,
        methods=require_methods(args.methods),
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
    )
    write_csv_stdout(GRID_FIELDS, rows)
    if args.strict and problems:
        first = problems[0]
        raise SystemExit(
            "metric grid incomplete: "
            f"method={first[0]} day={first[1]} kind={first[2]} "
            f"missing_sample={first[3]} duplicates={first[4]} file={first[5]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
