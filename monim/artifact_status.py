#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

from result_utils import METHODS, expected_grid, read_metric_pairs


def has_metric_pair(path: Path, lmbda: float, temp: float) -> bool:
    for found_lmbda, found_temp in read_metric_pairs(path):
        if math.isclose(found_lmbda, lmbda) and math.isclose(found_temp, temp):
            return True
    return False


def metric_pair_done(args) -> int:
    ok = has_metric_pair(args.ppl_file, args.lmbda, args.temp) and has_metric_pair(args.pwd_file, args.lmbda, args.temp)
    return 0 if ok else 1


def metric_grid_done(args) -> int:
    expected = expected_grid()
    ppl = read_metric_pairs(args.ppl_file)
    pwd = read_metric_pairs(args.pwd_file)
    ok = len(ppl) == len(expected) and set(ppl) == expected and len(pwd) == len(expected) and set(pwd) == expected
    return 0 if ok else 1


def size_entry(args) -> int:
    data = json.loads(args.size_json.read_text())
    value = data[args.dataset][args.ckpt_name][str(args.day)][args.param]
    print(int(value))
    return 0


def npy_count(args) -> int:
    keys = np.load(args.keys, mmap_mode="r")
    vals = np.load(args.vals, mmap_mode="r")
    if keys.shape[0] != vals.shape[0]:
        return 1
    print(keys.shape[0])
    return 0


def checkpoint_newer(args) -> int:
    if not args.ckpt.is_file() or args.ckpt.stat().st_size == 0:
        return 1
    try:
        input_mtimes = [path.stat().st_mtime for path in args.inputs]
    except OSError:
        return 1
    return 0 if input_mtimes and args.ckpt.stat().st_mtime >= max(input_mtimes) else 1


def method_params(args) -> int:
    key = args.method.lower()
    if key not in METHODS:
        raise SystemExit(f"Unknown method: {args.method}")
    print(METHODS[key].param.replace(",", " "))
    return 0


def method_uses_adapter(args) -> int:
    key = args.method.lower()
    if key not in METHODS:
        raise SystemExit(f"Unknown method: {args.method}")
    return 0 if METHODS[key].net_param is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("metric-pair-done")
    pair.add_argument("ppl_file", type=Path)
    pair.add_argument("pwd_file", type=Path)
    pair.add_argument("lmbda", type=float)
    pair.add_argument("temp", type=float)
    pair.set_defaults(func=metric_pair_done)

    grid = subparsers.add_parser("metric-grid-done")
    grid.add_argument("ppl_file", type=Path)
    grid.add_argument("pwd_file", type=Path)
    grid.set_defaults(func=metric_grid_done)

    size = subparsers.add_parser("size-entry")
    size.add_argument("size_json", type=Path)
    size.add_argument("dataset")
    size.add_argument("ckpt_name")
    size.add_argument("day", type=int)
    size.add_argument("param")
    size.set_defaults(func=size_entry)

    count = subparsers.add_parser("npy-count")
    count.add_argument("keys", type=Path)
    count.add_argument("vals", type=Path)
    count.set_defaults(func=npy_count)

    ckpt = subparsers.add_parser("checkpoint-newer")
    ckpt.add_argument("ckpt", type=Path)
    ckpt.add_argument("inputs", nargs="+", type=Path)
    ckpt.set_defaults(func=checkpoint_newer)

    params = subparsers.add_parser("method-params")
    params.add_argument("method")
    params.set_defaults(func=method_params)

    adapter = subparsers.add_parser("method-uses-adapter")
    adapter.add_argument("method")
    adapter.set_defaults(func=method_uses_adapter)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
