#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

from result_utils import (
    GRID_FIELDS,
    METHODS,
    SUMMARY_FIELDS,
    default_root,
    metric_grid_rows,
    method_output_dir,
    summarize_rows,
    write_csv_file,
)


def fail_if_strict(strict: bool, summary_problems: List[str], grid_problems: List[Tuple]) -> None:
    if not strict:
        return
    if summary_problems:
        raise SystemExit(
            f"Incomplete metric grids for {len(summary_problems)} day-method entries; first: {summary_problems[0]}"
        )
    if grid_problems:
        first = grid_problems[0]
        raise SystemExit(
            "metric grid incomplete: "
            f"method={first[0]} day={first[1]} kind={first[2]} "
            f"missing_sample={first[3]} duplicates={first[4]} file={first[5]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ckpt-name", default="gpt2-small")
    parser.add_argument("--dataset", default="daily")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir or root / "monim" / "final"
    fullmem = METHODS["fullmem"]
    monim = METHODS["monim"]

    fullmem_results, fullmem_summary_problems = summarize_rows(
        output_dir=method_output_dir(root, fullmem),
        methods=[fullmem],
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
        dataset=args.dataset,
    )
    monim_results, monim_summary_problems = summarize_rows(
        output_dir=method_output_dir(root, monim),
        methods=[monim],
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
        dataset=args.dataset,
    )
    results = sorted(fullmem_results + monim_results, key=lambda row: (int(row["day"]), row["method"]))

    fullmem_grid, fullmem_grid_problems = metric_grid_rows(
        output_dir=method_output_dir(root, fullmem),
        methods=[fullmem],
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
    )
    monim_grid, monim_grid_problems = metric_grid_rows(
        output_dir=method_output_dir(root, monim),
        methods=[monim],
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
    )
    metric_grid = sorted(fullmem_grid + monim_grid, key=lambda row: (row["method"], int(row["day"]), row["kind"]))

    fail_if_strict(
        args.strict,
        fullmem_summary_problems + monim_summary_problems,
        fullmem_grid_problems + monim_grid_problems,
    )

    results_path = out_dir / f"results_{args.begin}_{args.end}.csv"
    metric_grid_path = out_dir / f"metric_grid_{args.begin}_{args.end}.csv"
    write_csv_file(results_path, SUMMARY_FIELDS, results)
    write_csv_file(metric_grid_path, GRID_FIELDS, metric_grid)

    print(f"Results:     {results_path}")
    print(f"Metric grid: {metric_grid_path}")


if __name__ == "__main__":
    main()
