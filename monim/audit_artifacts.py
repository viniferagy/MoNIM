#!/usr/bin/env python3
import argparse
from pathlib import Path

from result_utils import AUDIT_FIELDS, artifact_audit_rows, default_root, write_csv_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--ckpt-name", default="gpt2-small")
    parser.add_argument("--dataset", default="daily")
    args = parser.parse_args()

    rows = artifact_audit_rows(
        root=args.root.resolve(),
        begin=args.begin,
        end=args.end,
        ckpt_name=args.ckpt_name,
        dataset=args.dataset,
    )
    write_csv_stdout(AUDIT_FIELDS, rows)


if __name__ == "__main__":
    main()
