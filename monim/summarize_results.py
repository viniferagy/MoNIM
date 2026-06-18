#!/usr/bin/env python3
import argparse
from pathlib import Path

from result_utils import SUMMARY_FIELDS, default_root, require_methods, summarize_rows, write_csv_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", default="monim")
    parser.add_argument("--ckpt-name", default="gpt2-small")
    parser.add_argument("--dataset", default="daily")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any metric file is missing")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir is not None else root / "output"
    methods = require_methods(args.methods)
    rows, problems = summarize_rows(
        output_dir=output_dir,
        methods=methods,
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
        dataset=args.dataset,
        day=args.day,
    )
    write_csv_stdout(SUMMARY_FIELDS, rows)
    if args.strict and problems:
        raise SystemExit(f"Incomplete metric grids for {len(problems)} day-method entries; first: {problems[0]}")


if __name__ == "__main__":
    main()
